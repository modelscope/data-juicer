from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import pyarrow
from jsonargparse import Namespace
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.data import DJDataset
from data_juicer.core.data.schema import Schema
from data_juicer.core.RayOperatorWrapper import Actor
from data_juicer.ops import Deduplicator, Filter, Mapper
from data_juicer.ops.base_op import TAGGING_OPS
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import is_remote_path
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.process_utils import calculate_np
from data_juicer.utils.webdataset_utils import _custom_default_decoder

ray = LazyLoader("ray")


def get_abs_path(path, dataset_dir):
    if is_remote_path(path):
        return path
    full_path = os.path.abspath(os.path.join(dataset_dir, path))
    if os.path.exists(full_path):
        return full_path
    else:
        return path


def convert_to_absolute_paths(samples, dataset_dir, path_keys):
    samples = samples.to_pydict()
    for key in path_keys:
        for idx in range(len(samples[key])):
            paths = samples[key][idx]
            if isinstance(paths, str):
                samples[key][idx] = get_abs_path(paths, dataset_dir)
            elif isinstance(paths, list):
                samples[key][idx] = [get_abs_path(item, dataset_dir) for item in paths]
    return pyarrow.Table.from_pydict(samples)


# TODO: check path for nestdataset
def set_dataset_to_absolute_path(dataset, dataset_path, cfg):
    """
    Set all the path in input data to absolute path.
    Checks dataset_dir and project_dir for valid paths.
    """
    path_keys = []
    columns = dataset.columns()
    for key in [
        cfg.get("video_key", "videos"),
        cfg.get("image_key", "images"),
        cfg.get("audio_key", "audios"),
    ]:
        if key in columns:
            path_keys.append(key)
    if len(path_keys) > 0:
        dataset_dir = os.path.dirname(dataset_path)
        logger.info(f"dataset_dir: {dataset_dir}")
        dataset = dataset.map_batches(
            partial(convert_to_absolute_paths, dataset_dir=dataset_dir, path_keys=path_keys),
            batch_format="pyarrow",
            zero_copy_batch=True,
        )
    return dataset


def preprocess_dataset(dataset: ray.data.Dataset, dataset_path, cfg) -> ray.data.Dataset:
    if dataset_path:
        dataset = set_dataset_to_absolute_path(dataset, dataset_path, cfg)
    return dataset


def get_num_gpus(op, op_proc):
    if not op.use_cuda():
        return 0
    proc_per_gpu = op_proc / cuda_device_count()
    return 1.0 / proc_per_gpu


def filter_batch(batch, filter_func):
    mask = pyarrow.array(filter_func(batch.to_pydict()))
    return batch.filter(mask)


class RayDataset(DJDataset):
    def __init__(self, dataset: ray.data.Dataset, dataset_path: str = None, cfg: Optional[Namespace] = None) -> None:
        self.data = preprocess_dataset(dataset, dataset_path, cfg)
        self.num_proc = getattr(cfg, "np", getattr(cfg, "num_proc", None)) if cfg else None

    def schema(self) -> Schema:
        """Get dataset schema.

        Returns:
            Schema: Dataset schema containing column names and types
        """
        if self.data is None or self.data.columns() is None:
            raise ValueError("Dataset is empty or not initialized")

        return Schema.from_ray_schema(self.data.schema())

    def get(self, k: int) -> List[Dict[str, Any]]:
        """Get k rows from the dataset."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        if k == 0:
            return []

        k = min(k, self.data.count())
        return list(self.data.limit(k).take())

    def get_column(self, column: str, k: Optional[int] = None) -> List[Any]:
        """Get column values from Ray dataset.

        Args:
            column: Name of the column to retrieve
            k: Optional number of rows to return. If None, returns all rows

        Returns:
            List of values from the specified column

        Raises:
            KeyError: If column doesn't exist
            ValueError: If k is negative
        """
        if self.data is None or self.data.columns() is None or column not in self.data.columns():
            raise KeyError(f"Column '{column}' not found in dataset")

        if k is not None:
            if k < 0:
                raise ValueError(f"k must be non-negative, got {k}")
            if k == 0:
                return []
            k = min(k, self.data.count())
            return [row[column] for row in self.data.limit(k).take()]

        return [row[column] for row in self.data.take()]

    def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]
        for op in operators:
            self._run_single_op(op)
            self.data = self.data.materialize()
        return self

    def process_parallel(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        """
        Process the dataset in parallel using multiple operators.
        This method creates actors for each operator and processes the dataset in a streaming manner.
        """
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]

        # Add meta and stats columns if needed
        add_meta = False
        add_stats = False
        for op in operators:
            columns = self.data.columns()
            if op._name in TAGGING_OPS.modules and Fields.meta not in self.data.columns():
                add_meta = True
            if isinstance(op, Filter):
                if Fields.stats not in columns:
                    add_stats = True

        if add_meta:

            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.meta, [new_column_data])
                print("new_table:", new_table)
                return new_table

            self.data = self.data.map_batches(process_batch_arrow, batch_format="pyarrow")

        if add_stats:

            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.stats, [new_column_data])
                print("new_table:", new_table)
                return new_table

            self.data = self.data.map_batches(process_batch_arrow, batch_format="pyarrow")

        # Step 1: Create actors for all operators
        actors = {}
        for op in operators:
            op_proc = (
                1
                if op.use_cuda()
                else calculate_np(op._name, op.mem_required, op.cpu_required, self.num_proc, op.use_cuda())
            )
            actor_num = op_proc
            actors[op._name] = []

            for _ in range(actor_num):
                actor = Actor.options(
                    name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",
                    num_gpus=op.gpu_required if op.use_cuda() else 0,
                    num_cpus=op.cpu_required,
                ).remote(op)

                if op.use_cuda():
                    actor.load_model.remote()

                actors[op._name].append(actor)

            logger.info(f"Operator {op._name} has {len(actors[op._name])} actor(s).")

        # Step 2: Set batch size for each operator
        batch_sizes = {op._name: op.batch_size if hasattr(op, "batch_size") else 1 for op in operators}

        logger.info(f"Batch sizes per operator: {batch_sizes}")

        # Step 3: Process single operator streaming
        if len(operators) == 1:
            return self._process_single_operator_streaming(
                operators[0], actors[operators[0]._name], batch_sizes[operators[0]._name]
            )

        # Step 4: Create queues for each actor and termination counters
        actor_queues = {}
        termination_counters = {}
        for op in operators:
            actor_queues[op._name] = []
            termination_counters[op._name] = {"count": 0, "lock": threading.Lock(), "total": len(actors[op._name])}
            for i, actor in enumerate(actors[op._name]):
                actor_queues[op._name].append(queue.Queue(maxsize=50))

        final_results = []
        result_lock = threading.Lock()

        # Step 5: Start processing threads for each operator's actors
        threads = []
        for idx, op in enumerate(operators):
            for i, actor in enumerate(actors[op._name]):
                thread = threading.Thread(
                    target=self._process_actor_streaming,
                    args=(
                        idx,
                        op,
                        actor,
                        i,
                        actor_queues,
                        operators,
                        final_results,
                        result_lock,
                        batch_sizes[op._name],
                        termination_counters,
                    ),
                    name=f"actor_{op._name}_{i}",
                    daemon=True,
                )
                thread.start()
                threads.append(thread)

        # Step 6: Data distributor function to distribute data to actors
        def data_distributor():
            first_op = operators[0]
            first_op_queues = actor_queues[first_op._name]
            actor_index = 0
            row_counter = 0  # Initialize row counter

            try:
                for batch in self.data.iter_batches(batch_size=1, batch_format="pyarrow"):
                    for row_idx in range(len(batch)):
                        row_data = {col: batch[col][row_idx].as_py() for col in batch.column_names}
                        row_data["_row_id"] = row_counter
                        row_counter += 1

                        # distribute data to actors in a round-robin manner
                        target_queue = first_op_queues[actor_index % len(first_op_queues)]
                        target_queue.put(row_data)
                        actor_index += 1

            except Exception as e:
                logger.error(f"Error in data distributor: {e}")
            finally:
                for actor_queue in first_op_queues:
                    actor_queue.put(None)

        # start data distributor thread
        distributor_thread = threading.Thread(target=data_distributor, daemon=True)
        distributor_thread.start()

        distributor_thread.join()

        # wait for all processing threads to finish
        for thread in threads:
            thread.join()
        if final_results:

            self.data = ray.data.from_items(final_results)

        return self

    def _process_actor_streaming(
        self,
        op_idx,
        op,
        actor,
        actor_id,
        actor_queues,
        operators,
        final_results,
        result_lock,
        batch_size,
        termination_counters,
    ):
        """Process data for a single operator actor in a streaming manner."""
        op_name = op._name
        input_queue = actor_queues[op_name][actor_id]

        next_op_queues = None
        if op_idx + 1 < len(operators):
            next_op_name = operators[op_idx + 1]._name
            next_op_queues = actor_queues[next_op_name]

        logger.info(f"Starting streaming processor for {op_name} actor {actor_id}")
        processed_count = 0
        batch_buffer = []
        next_actor_index = 0

        # data flow logging function
        def log_data_flow(row_id, action, start_time=None):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if action == "start":
                logger.info(f"[DataFlow] Row {row_id} | {op_name}_actor_{actor_id} | START | {timestamp}")
            elif action == "end":
                duration = time.time() - start_time
                logger.info(
                    f"[DataFlow] Row {row_id} | {op_name}_actor_{actor_id} | END | {timestamp} | Duration: {duration:.3f} s"
                )

        while True:
            try:
                data_item = input_queue.get(timeout=5.0)

                if data_item is None:
                    if batch_buffer:
                        results_count = self._process_and_forward_batch(
                            op,
                            actor,
                            batch_buffer,
                            next_op_queues,
                            final_results,
                            result_lock,
                            next_actor_index,
                            log_data_flow,
                        )
                        next_actor_index += results_count

                    # update termination counter
                    with termination_counters[op_name]["lock"]:
                        termination_counters[op_name]["count"] += 1
                        current_count = termination_counters[op_name]["count"]
                        total_actors = termination_counters[op_name]["total"]

                    # notify the downstream only when all actors have received None
                    if current_count >= total_actors and next_op_queues:
                        for q in next_op_queues:
                            q.put(None)

                    break

                row_id = data_item.get("_row_id", "unknown")
                start_time = time.time()
                log_data_flow(row_id, "start", start_time)

                batch_buffer.append((data_item, start_time, row_id))

                if len(batch_buffer) >= batch_size or not op.is_batched_op():
                    results_count = self._process_and_forward_batch(
                        op,
                        actor,
                        batch_buffer,
                        next_op_queues,
                        final_results,
                        result_lock,
                        next_actor_index,
                        log_data_flow,
                    )
                    next_actor_index += results_count
                    processed_count += len(batch_buffer)
                    batch_buffer = []

            except queue.Empty:
                if batch_buffer:
                    results_count = self._process_and_forward_batch(
                        op,
                        actor,
                        batch_buffer,
                        next_op_queues,
                        final_results,
                        result_lock,
                        next_actor_index,
                        log_data_flow,
                    )
                    next_actor_index += results_count
                    processed_count += len(batch_buffer)
                    batch_buffer = []
                continue
            except Exception as e:
                logger.error(f"Error in {op_name} actor {actor_id}: {e}")
                break

        logger.info(f"Streaming processor for {op_name} actor {actor_id} completed, processed {processed_count} items")

    def _process_batch(self, op, actor, batch_data, final_results, result_lock):
        """Process a batch of data with the given operator and actor."""
        if not batch_data:
            return

        try:
            if len(batch_data) == 1:
                future = self._submit_to_actor(op, actor, batch_data[0])
                results = ray.get(future)
            else:
                futures = [self._submit_to_actor(op, actor, item) for item in batch_data]
                results = ray.get(futures)
                # flatten results
                flattened_results = []
                for result in results:
                    if isinstance(result, list):
                        flattened_results.extend(result)
                    elif result is not None:
                        flattened_results.append(result)
                results = flattened_results

            with result_lock:
                if isinstance(results, list):
                    final_results.extend(results)
                elif results is not None:
                    final_results.append(results)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

    def _submit_to_actor(self, op, actor, data_item):
        """Submit a single data item to the actor for processing."""
        if isinstance(op, Mapper):
            if op.use_cuda():
                return (
                    actor.mapper_cuda_batched.remote(self.transform_to_2d_format(data_item))
                    if op.is_batched_op()
                    else actor.mapper_cuda.remote(data_item)
                )
            else:
                return actor.mapper_cpu.remote(data_item)

        elif isinstance(op, Filter):
            if op.use_cuda():
                return (
                    actor.filter_cuda_batched.remote(data_item)
                    if op.is_batched_op()
                    else actor.filter_cuda_single.remote(data_item)
                )
            else:
                return (
                    actor.filter_cpu_batched.remote(self.transform((data_item)))
                    if op.is_batched_op()
                    else actor.filter_cpu_single.remote(data_item)
                )

    def _process_and_forward_batch(
        self,
        op,
        actor,
        batch_data_with_metadata,
        next_op_queues,
        final_results,
        result_lock,
        next_actor_index,
        log_data_flow,
    ):
        """Process batch data and forward to downstream with data flow tracking"""
        if not batch_data_with_metadata:
            return 0
        # separate the data, start time, and line number
        batch_data = [item[0] for item in batch_data_with_metadata]
        start_times = [item[1] for item in batch_data_with_metadata]
        row_ids = [item[2] for item in batch_data_with_metadata]

        try:
            if len(batch_data) == 1:
                future = self._submit_to_actor(op, actor, batch_data[0])
                results = ray.get(future)
            else:
                futures = [self._submit_to_actor(op, actor, item) for item in batch_data]
                results = ray.get(futures)
                # flatten results
                flattened_results = []
                for result in results:
                    if isinstance(result, list):
                        flattened_results.extend(result)
                    elif result is not None:
                        flattened_results.append(result)
                results = flattened_results

            valid_results = []
            if isinstance(op, Mapper):
                if isinstance(results, list):
                    valid_results = results
                elif results is not None:
                    valid_results = [results]
            elif isinstance(op, Filter):
                if results:
                    if isinstance(results, list):
                        valid_results = results
                    else:
                        valid_results = [results]

            for row_id, start_time in zip(row_ids, start_times):
                log_data_flow(row_id, "end", start_time)

            if next_op_queues and valid_results:
                # distribute results to next operator queues
                for i, result in enumerate(valid_results):
                    try:
                        # add row_id to result if needed
                        if isinstance(result, dict):
                            result["_row_id"] = row_ids[i % len(row_ids)]

                        target_queue_idx = (next_actor_index + i) % len(next_op_queues)
                        next_op_queues[target_queue_idx].put(result)
                    except Exception as e:
                        logger.error(f"Error forwarding result to downstream queue: {e}")
            elif not next_op_queues and valid_results:
                with result_lock:
                    final_results.extend(valid_results)

            return len(valid_results)

        except Exception as e:
            for row_id, start_time in zip(row_ids, start_times):
                log_data_flow(row_id, "end", start_time)
            logger.error(f"Error processing and forwarding batch: {e}")
            return 0

    def _process_single_operator_streaming(self, op, op_actors, batch_size):
        """Stream processing for a single operator."""
        final_results = []
        result_lock = threading.Lock()

        # create an independent queue for each actor.
        actor_queues = [queue.Queue(maxsize=50) for _ in op_actors]

        # start the actor processing threads
        threads = []
        for i, actor in enumerate(op_actors):
            thread = threading.Thread(
                target=self._process_single_actor,
                args=(op, actor, actor_queues[i], final_results, result_lock, batch_size),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        actor_index = 0
        for batch in self.data.iter_batches(batch_size=1, batch_format="pyarrow"):
            for row_idx in range(len(batch)):
                row_data = {col: batch[col][row_idx].as_py() for col in batch.column_names}
                target_queue = actor_queues[actor_index % len(actor_queues)]
                target_queue.put(row_data)
                actor_index += 1

        # notify all Actors to terminate.
        for actor_queue in actor_queues:
            actor_queue.put(None)

        for thread in threads:
            thread.join()

        if final_results:
            self.data = ray.data.from_items(final_results)

        return self

    def _process_single_actor(self, op, actor, input_queue, final_results, result_lock, batch_size):
        """
        Process data for a single actor in a streaming manner.
        This function continuously reads data from the input queue, processes it, and stores the results.
        """
        batch_buffer = []
        processed_count = 0

        while True:
            try:
                data_item = input_queue.get(timeout=30.0)
                if data_item is None:
                    if batch_buffer:
                        self._process_batch(op, actor, batch_buffer, final_results, result_lock)
                    break

                batch_buffer.append(data_item)

                # process when the batch is full or during batch processing operations.
                if len(batch_buffer) >= batch_size or not op.is_batched_op():
                    processed_batch_len = len(batch_buffer)
                    self._process_batch(op, actor, batch_buffer, final_results, result_lock)
                    batch_buffer = []
                    processed_count += processed_batch_len

            except queue.Empty:
                if batch_buffer:
                    self._process_batch(op, actor, batch_buffer, final_results, result_lock)
                    batch_buffer = []
                continue
            except Exception as e:
                logger.error(f"Error in single actor processing: {e}")
                break
        logger.info(f"Single actor completed, processed {processed_count} items")

    def transform(self, data):

        if not isinstance(data.get("text"), list):
            data["text"] = [data["text"]]

        if not isinstance(data.get("__dj__stats__"), list):
            data["__dj__stats__"] = [data["__dj__stats__"]]

        return data

    def transform_to_2d_format(self, data):
        """
        Transform data to 2D format for processing.
        """
        if "__dj__source_file__" not in data:
            if "videos" not in data:
                raise ValueError(
                    "The '__dj__source_file__' field is missing in the data and cannot be inferred from the 'videos' field."
                )
            data["__dj__source_file__"] = data["videos"]

        source_files = data["__dj__source_file__"]

        unique_sources = list(dict.fromkeys(source_files))

        source_to_indices = {}
        for source in unique_sources:
            source_to_indices[source] = [i for i, s in enumerate(source_files) if s == source]

        transformed_data = {}
        for field_name, field_value in data.items():
            if field_name == "__dj__source_file__":

                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    transformed_data[field_name].append([source] * len(indices))
            elif isinstance(field_value, list):

                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    group_data = [field_value[i] for i in indices]
                    transformed_data[field_name].append(group_data)
            elif isinstance(field_value, dict):

                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    group_dict = {}
                    for key, values in field_value.items():
                        if isinstance(values, list):
                            group_dict[key] = [values[i] for i in indices]
                        else:
                            group_dict[key] = [values] * len(indices)
                    transformed_data[field_name].append(group_dict)
            elif isinstance(field_value, str):
                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    transformed_data[field_name].append(field_value)
            else:

                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    transformed_data[field_name].append(field_value)

        return transformed_data

    def _run_single_op(self, op):
        op_proc = calculate_np(op._name, op.mem_required, op.cpu_required, self.num_proc, op.use_cuda())
        num_gpus = get_num_gpus(op, op_proc)

        if op._name in TAGGING_OPS.modules and Fields.meta not in self.data.columns():

            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.meta, [new_column_data])
                return new_table

            self.data = self.data.map_batches(process_batch_arrow, batch_format="pyarrow")

        try:
            batch_size = getattr(op, "batch_size", 1) if op.is_batched_op() else 1
            if isinstance(op, Mapper):
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=None,
                        fn_constructor_kwargs=op_kwargs,
                        batch_size=batch_size,
                        num_gpus=num_gpus,
                        concurrency=op_proc,
                        batch_format="pyarrow",
                    )
                else:
                    self.data = self.data.map_batches(
                        op.process, batch_size=batch_size, batch_format="pyarrow", num_gpus=num_gpus
                    )
            elif isinstance(op, Filter):
                columns = self.data.columns()
                if Fields.stats not in columns:

                    def process_batch_arrow(table: pyarrow.Table):
                        new_column_data = [{} for _ in range(len(table))]
                        new_talbe = table.append_column(Fields.stats, [new_column_data])
                        return new_talbe

                    self.data = self.data.map_batches(process_batch_arrow, batch_format="pyarrow")
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=None,
                        fn_constructor_kwargs=op_kwargs,
                        batch_size=batch_size,
                        num_gpus=num_gpus,
                        concurrency=op_proc,
                        batch_format="pyarrow",
                    )
                else:
                    self.data = self.data.map_batches(
                        op.compute_stats, batch_size=batch_size, batch_format="pyarrow", num_gpus=num_gpus
                    )
                if op.stats_export_path is not None:
                    self.data.write_json(op.stats_export_path, force_ascii=False)
                if op.is_batched_op():
                    self.data = self.data.map_batches(
                        partial(filter_batch, filter_func=op.process),
                        batch_format="pyarrow",
                        batch_size=batch_size,
                        num_gpus=num_gpus,
                        zero_copy_batch=True,
                    )
                else:
                    self.data = self.data.filter(op.process)
            elif isinstance(op, Deduplicator):
                self.data = op.run(self.data)
            else:
                logger.error("Ray executor only support Filter and Mapper OPs for now")
                raise NotImplementedError
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op._name}].")
            import traceback

            traceback.print_exc()
            exit(1)

    @classmethod
    def read(cls, data_format: str, paths: Union[str, List[str]]) -> RayDataset:
        if data_format in {"json", "jsonl"}:
            return RayDataset.read_json(paths)
        elif data_format == "webdataset":
            return RayDataset.read_webdataset(paths)
        elif data_format in {
            "parquet",
            "images",
            "parquet_bulk",
            "csv",
            "text",
            "avro",
            "numpy",
            "tfrecords",
            "binary_files",
            "lance",
        }:
            return getattr(ray.data, f"read_{data_format}")(paths)

    @classmethod
    def read_json(cls, paths: Union[str, List[str]]) -> RayDataset:
        # Note: a temp solution for reading json stream
        # TODO: replace with ray.data.read_json_stream once it is available
        import pyarrow.json as js

        try:
            js.open_json
            return read_json_stream(paths)
        except AttributeError:
            return ray.data.read_json(paths)

    @classmethod
    def read_webdataset(cls, paths: Union[str, List[str]]) -> RayDataset:
        return ray.data.read_webdataset(paths, decoder=partial(_custom_default_decoder, format="PIL"))

    def to_list(self) -> list:
        return self.data.to_pandas().to_dict(orient="records")


class JSONStreamDatasource(ray.data.read_api.JSONDatasource):
    """
    A temp Datasource for reading json stream.

    Note:

        Depends on a customized `pyarrow` with `open_json` method.
    """

    def _read_stream(self, f: "pyarrow.NativeFile", path: str):
        from pyarrow.json import open_json

        try:
            reader = open_json(
                f,
                read_options=self.read_options,
                **self.arrow_json_args,
            )
            schema = None
            while True:
                try:
                    batch = reader.read_next_batch()
                    table = pyarrow.Table.from_batches([batch], schema=schema)
                    if schema is None:
                        schema = table.schema
                    yield table
                except StopIteration:
                    return
        except pyarrow.lib.ArrowInvalid as e:
            raise ValueError(f"Failed to read JSON file: {path}.") from e


def read_json_stream(
    paths: Union[str, List[str]],
    *,
    filesystem: Optional["pyarrow.fs.FileSystem"] = None,
    parallelism: int = -1,
    ray_remote_args: Dict[str, Any] = None,
    arrow_open_stream_args: Optional[Dict[str, Any]] = None,
    meta_provider=None,
    partition_filter=None,
    partitioning=ray.data.read_api.Partitioning("hive"),
    include_paths: bool = False,
    ignore_missing_paths: bool = False,
    shuffle: Union[Literal["files"], None] = None,
    file_extensions: Optional[List[str]] = ["json", "jsonl"],
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
    **arrow_json_args,
) -> ray.data.Dataset:
    if meta_provider is None:
        meta_provider = ray.data.read_api.DefaultFileMetadataProvider()

    datasource = JSONStreamDatasource(
        paths,
        arrow_json_args=arrow_json_args,
        filesystem=filesystem,
        open_stream_args=arrow_open_stream_args,
        meta_provider=meta_provider,
        partition_filter=partition_filter,
        partitioning=partitioning,
        ignore_missing_paths=ignore_missing_paths,
        shuffle=shuffle,
        include_paths=include_paths,
        file_extensions=file_extensions,
    )
    return ray.data.read_datasource(
        datasource,
        parallelism=parallelism,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )
