from __future__ import annotations

import os
from functools import partial
import queue
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Union
import uuid
import numpy
from data_juicer.core.ray_actor import Actor
import pyarrow
from jsonargparse import Namespace
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.data import DJDataset
from data_juicer.core.data.schema import Schema
from data_juicer.ops import Deduplicator, Filter, Mapper
from data_juicer.ops.base_op import TAGGING_OPS
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import is_remote_path
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.process_utils import calculate_np
from data_juicer.utils.webdataset_utils import _custom_default_decoder
import ray
ray = LazyLoader("ray")
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.data import from_items

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
        # self.gpu_pg = placement_group([{"CPU": 16, "GPU": 2}], strategy="STRICT_SPREAD")
        # ray.get(self.gpu_pg.ready())

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

    def process1(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]
        for op in operators:
            self._run_single_op(op)
            self.data = self.data.materialize()
        return self
    
    def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]

        # 添加meta和stats列（如果需要）
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
                return new_table

            self.data = self.data.map_batches(process_batch_arrow, batch_format="pyarrow")

        if add_stats:
            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.stats, [new_column_data])
                return new_table

            self.data = self.data.map_batches(process_batch_arrow, batch_format="pyarrow")

        # 创建所有operator的actors（保持原有逻辑不变）
        actors = {}

        for idx, op in enumerate(operators):
            if op.use_cuda():
                op_proc = 1
            else:
                op_proc = calculate_np(op._name, op.mem_required, op.cpu_required, self.num_proc, op.use_cuda())
            
            actors[op._name] = []

            actor_num = min(op_proc, self.data.count())

            if op.use_cuda():
                print(f"{op._name} allocate {op.gpu_required} GPUs.")
                for _ in range(actor_num):
                    actor = Actor.options(
                        name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",
                        num_gpus=op.gpu_required,
                        num_cpus=op.cpu_required,
                    ).remote(op)
                    actor.load_model.remote()
                    actors[op._name].append(actor)
            else:
                print(f"{op._name} allocate in CPU.")
                for _ in range(actor_num):
                    actor = Actor.options(
                        name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",
                        num_gpus=0,
                        num_cpus=op.cpu_required,
                    ).remote(op)
                    actors[op._name].append(actor)

        # 打印所有actor信息
        for op_name, actor_list in actors.items():
            logger.info(f"Operator {op_name} has the following actors:")
            for i, actor in enumerate(actor_list):
                logger.info(f"  Actor {i}: {actor._ray_actor_id.hex()[:6]}")

        # 如果只有一个operator，直接处理
        if len(operators) == 1:
            return self._process_single_operator(operators[0], actors[operators[0]._name])

        # 创建operator队列（从第二个operator开始）
        op_buckets = {op._name: queue.Queue(maxsize=100) for op in operators[1:]}  # 添加maxsize防止内存爆炸

        # 存储最终结果
        final_results = []
        result_lock = threading.Lock()

        # 启动后续operator的处理线程（先启动消费者）
        threads = []
        for idx, op in enumerate(operators[1:], start=1):
            for actor in actors[op._name]:
                thread = threading.Thread(
                    target=self._process_operator,
                    args=(idx, op, actor, op_buckets, actors, operators, final_results, result_lock),
                    name=f"processor_{op._name}_{actor._ray_actor_id.hex()[:6]}",
                    daemon=True
                )
                thread.start()
                threads.append(thread)

        # 动态调整batch_size以控制内存使用
        estimated_row_count = self.data.count()

        # 为每个operator设置batch_size为其actor数量
        batch_sizes = {
            op._name: max(1, estimated_row_count // len(actors[op._name]))
            for op in operators
        }

        # 使用iter_batches并行处理第一个operator的数据
        first_op = operators[0]
        first_op_actors = actors[first_op._name]
        actor_index = 0
        
        try:
            for batch in self.data.iter_batches(
                batch_size=batch_sizes[first_op._name],
                batch_format="pyarrow"
            ):
                # 将batch转换为行数据进行处理
                for row_idx in range(len(batch)):
                    # 提取单行数据
                    row_data = {}
                    for col_name in batch.column_names:
                        col_data = batch.column(col_name)
                        row_data[col_name] = col_data[row_idx].as_py()
                    
                    # 选择actor进行负载均衡
                    actor = first_op_actors[actor_index % len(first_op_actors)]
                    actor_index += 1
                    
                    # 异步处理数据
                    future = self._submit_to_actor(first_op, actor, row_data)
                    result = ray.get(future)
                    
                    # 将结果放入第二个operator的队列
                    if len(operators) > 1:
                        op_buckets[operators[1]._name].put(result)
                    else:
                        with result_lock:
                            if isinstance(result, list):
                                final_results.extend(result)
                            else:
                                final_results.append(result)

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

        # 添加结束标记到第二个operator的队列
        if len(operators) > 1:
            for _ in range(len(actors[operators[1]._name])):
                op_buckets[operators[1]._name].put(None)

            # 等待所有线程完成
            for thread in threads:
                thread.join()

        # 返回最终结果
        if final_results:
            self.data = from_items(final_results)
        return self
    
    def _process_single_operator(self, op, op_actors):
        """处理只有一个operator的情况"""
        final_results = []
        actor_index = 0
        
        # 动态调整batch_size
        estimated_row_count = self.data.count()
        batch_size = max(1, min(1000, estimated_row_count // (len(op_actors) * 10)))
        
        for batch in self.data.iter_batches(
            batch_size=batch_size,
            batch_format="pyarrow"
        ):
            # 将batch转换为行数据进行处理
            for row_idx in range(len(batch)):
                # 提取单行数据
                row_data = {}
                for col_name in batch.column_names:
                    col_data = batch.column(col_name)
                    row_data[col_name] = col_data[row_idx].as_py()
                
                # 选择actor进行负载均衡
                actor = op_actors[actor_index % len(op_actors)]
                actor_index += 1
                
                # 处理数据
                future = self._submit_to_actor(op, actor, row_data)
                result = ray.get(future)
                
                if isinstance(result, list):
                    final_results.extend(result)
                elif result is not None:
                    final_results.append(result)
        
        if final_results:
            self.data = from_items(final_results)
        return self

    def _submit_to_actor(self, op, actor, data_item):
        """提交数据到actor进行处理"""
        if isinstance(op, Mapper):
            if op.use_cuda():
                if op.is_batched_op():
                    data_item = self.transform_to_2d_format(data_item)
                    return actor.mapper_cuda_batched.remote(data_item)
                else:
                    return actor.mapper_cuda.remote(data_item)
            else:
                return actor.mapper_cpu.remote(data_item)
        
        elif isinstance(op, Filter):
            if op.use_cuda():
                if op.is_batched_op():
                    return actor.filter_cuda_batched.remote(data_item)
                else:
                    return actor.filter_cuda_single.remote(data_item)
            else:
                if op.is_batched_op():
                    return actor.filter_cpu_batched.remote(data_item)
                else:
                    return actor.filter_cpu_single.remote(data_item)

    def _process_operator(self, op_idx, op, actor, op_buckets, actors, operators, final_results, result_lock):
        op_name = op._name
        input_queue = op_buckets[op_name]
        
        # 确定输出队列
        if op_idx + 1 < len(operators):
            output_queue = op_buckets[operators[op_idx + 1]._name]
        else:
            output_queue = None
        
        logger.info(f"Starting processor for {op_name} actor {actor._ray_actor_id.hex()[:6]}")

        start_time = time.time()
        processed_count = 0

        while True:
            try:
                # 从输入队列获取数据
                data_item = input_queue.get(timeout=30.0)  # 增加timeout
                
                # 检查结束标记
                if data_item is None:
                    if output_queue:
                        # 向下一个operator传递结束标记
                        for _ in range(len(actors[operators[op_idx + 1]._name])):
                            output_queue.put(None)
                    break
                
                # 处理数据
                future = self._submit_to_actor(op, actor, data_item)
                results = ray.get(future)
                processed_count += 1

                if isinstance(op, Mapper):
                    if output_queue:
                        output_queue.put(results)
                    else:
                        with result_lock:
                            if isinstance(results, list):
                                final_results.extend(results)
                            else:
                                final_results.append(results)
                
                elif isinstance(op, Filter):
                    if results:
                        if output_queue:
                            if isinstance(results, list):
                                for result in results:
                                    output_queue.put(result)
                            else:
                                output_queue.put(results)
                        else:
                            with result_lock:
                                if isinstance(results, list):
                                    final_results.extend(results)
                                else:
                                    final_results.append(results)

                # 标记任务完成
                input_queue.task_done()

            except queue.Empty:
                logger.warning(f"{op_name} actor {actor._ray_actor_id.hex()[:6]} queue timeout, processed {processed_count} items")
                continue
            except Exception as e:
                logger.error(f"Error in {op_name} actor {actor._ray_actor_id.hex()[:6]}: {e}")
                input_queue.task_done()
                break

        end_time = time.time()
        logger.info(f"Processor for {op_name} actor {actor._ray_actor_id.hex()[:6]} completed in {end_time - start_time:.2f} seconds, processed {processed_count} items")
    def transform_to_2d_format(self, data):
        """
        将第二种格式的数据转换为第一种嵌套格式
        根据 __dj__source_file__ 的唯一值来分组所有字段
        """
        # print("data before trans", data)
        if '__dj__source_file__' not in data:
            raise ValueError("数据中必须包含 '__dj__source_file__' 字段")
        
        source_files = data['__dj__source_file__']
        
        # 获取唯一的源文件并保持顺序
        unique_sources = list(dict.fromkeys(source_files))
        
        # 为每个唯一源文件创建索引映射
        source_to_indices = {}
        for source in unique_sources:
            source_to_indices[source] = [i for i, s in enumerate(source_files) if s == source]
        
        # 初始化转换后的数据结构
        transformed_data = {}
        
        # 遍历原数据的所有字段
        for field_name, field_value in data.items():
            if field_name == '__dj__source_file__':
                # 特殊处理 __dj__source_file__ 字段
                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    transformed_data[field_name].append([source] * len(indices))
            elif isinstance(field_value, list):
                # 处理列表类型的字段
                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    group_data = [field_value[i] for i in indices]
                    transformed_data[field_name].append(group_data)
            elif isinstance(field_value, dict):
                # 处理字典类型的字段
                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    group_dict = {}
                    for key, values in field_value.items():
                        if isinstance(values, list):
                            group_dict[key] = [values[i] for i in indices]
                        else:
                            # 如果值不是列表，则重复该值
                            group_dict[key] = [values] * len(indices)
                    transformed_data[field_name].append(group_dict)
            elif isinstance(field_value, str):
                # 处理字符串类型的字段
                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    # 对于字符串，为每个组重复该字符串
                    transformed_data[field_name].append(field_value)
            else:
                # 处理其他类型的字段
                transformed_data[field_name] = []
                for source in unique_sources:
                    indices = source_to_indices[source]
                    # 为每个组重复该值
                    transformed_data[field_name].append(field_value)
        # print("data after trans", transformed_data)
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
