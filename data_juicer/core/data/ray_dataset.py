from __future__ import annotations

import itertools
import math
import os
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import pyarrow
from jsonargparse import Namespace
from loguru import logger

from data_juicer.core.data import DJDataset
from data_juicer.core.data.schema import Schema
from data_juicer.ops import Deduplicator, Filter, Mapper
from data_juicer.ops.base_op import DEFAULT_BATCH_SIZE, TAGGING_OPS
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import is_remote_path
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.resource_utils import cuda_device_count
from data_juicer.utils.webdataset_utils import _custom_default_decoder

ray = LazyLoader("ray")
_OPS_MEMORY_LIMIT_FRACTION = 0.7


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
            batch_size=DEFAULT_BATCH_SIZE,
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


def find_optimal_concurrency(resource_ratios, total_resource):
    """
    Search for the optimal concurrency allocation to achieve the
    highest total resource utilization and the most balanced processing capacity.

    Args:
        resource_ratios (list[float]): List of single-process resource ratios for each operator
        total_resource (float): Total resource

    Return:
        tuple: (list of optimal concurrency, total resource usage, standard deviation of processing capacity)
        If there is no valid combination, return (None, 0, 0)
    """
    n = len(resource_ratios)
    if n == 0:
        return (None, 0, 0)

    sum_r_squared = sum(r * r for r in resource_ratios)
    if sum_r_squared == 0:
        return (None, 0, 0)

    c_floats = [(total_resource * r) / sum_r_squared for r in resource_ratios]

    # generate candidate concurrency
    candidates = []
    for cf in c_floats:
        floor_cf = math.floor(cf)
        ceil_cf = math.ceil(cf)
        possible = set()
        if floor_cf >= 1:
            possible.add(floor_cf)
        possible.add(ceil_cf)
        possible = [max(1, v) for v in possible]
        candidates.append(sorted(list(set(possible))))

    # traverse all combinations
    best_combination = None
    max_resource_usage = 0
    min_std = float("inf")

    for combo in itertools.product(*candidates):
        total_used = sum(c * r for c, r in zip(combo, resource_ratios))
        if total_used > total_resource:
            continue

        # calculate the standard deviation of processing capacity
        processing_powers = [c / r for c, r in zip(combo, resource_ratios)]
        mean = sum(processing_powers) / n
        variance = sum((x - mean) ** 2 for x in processing_powers) / n
        std = math.sqrt(variance)

        # update the optimal solution (priority resource utilization, suboptimal standard deviation)
        if total_used > max_resource_usage:
            max_resource_usage = total_used
            best_combination = combo
            min_std = std
        elif total_used == max_resource_usage and std < min_std:
            best_combination = combo
            min_std = std

    return (
        list(best_combination) if best_combination else None,
        max_resource_usage,
        min_std if best_combination else 0,
    )


class RayDataset(DJDataset):
    def __init__(self, dataset: ray.data.Dataset, dataset_path: str = None, cfg: Optional[Namespace] = None) -> None:
        self.data = preprocess_dataset(dataset, dataset_path, cfg)

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

    @staticmethod
    def set_resource_for_ops(operators):
        """
        Automatically calculates optimal concurrency for Ray Data operator.
        This function handles both task and actor based operators, considering
        resource requirements and user specifications. The computation follows Ray Data's
        concurrency semantics while optimizing resource utilization.

        Key Concepts:
        - Resource Ratio: Individual operator's resource requirement (GPU/CPU/memory)
            compared to total cluster resources, using max(cpu_ratio, gpu_ratio, adjusted_mem_ratio)
        - Fixed Allocation: Portion of resources reserved by operators with user-specified num_proc
        - Dynamic Allocation: Remaining resources distributed among auto-scaling operators

        Design Logic:
        1. User Specification Priority:
            - If user provides concurrency setting, directly return it
            - Applies to both task and actor based operators
        2. Task Operators (equivalent to a cpu operator in dj):
            a. When unspecified: Return None to let Ray determine implicitly
            b. Auto-calculation: Returns maximum concurrency based on available
               resources and operator requirements
        3. Actor Operators (equivalent to a gpu operator in dj):
            a. Mandatory concurrency - set required gpus to 1 if unspecified, and then refer to the following `b`
               to calculate automatically based on this setting
            b. Auto-calculation returns tuple (min_concurrency, max_concurrency):
                i. Minimum: Ensures baseline resource allocation in remaining resources
                    when all operators are active simultaneously (proportionally)
                ii. Maximum: Allows full utilization of remaining resources by single
                    operator when others are idle
        """
        from data_juicer.utils.ray_utils import (
            ray_available_gpu_memories,
            ray_available_memories,
            ray_cpu_count,
            ray_gpu_count,
        )
        from data_juicer.utils.resource_utils import is_cuda_available

        cuda_available = is_cuda_available()

        total_cpu = ray_cpu_count()
        total_gpu = ray_gpu_count()

        available_mem = sum(ray_available_memories()) * _OPS_MEMORY_LIMIT_FRACTION / 1024  # Convert MB to GB
        available_gpu_mem = sum(ray_available_gpu_memories()) * _OPS_MEMORY_LIMIT_FRACTION / 1024  # Convert MB to GB

        resource_configs = {}

        for op in operators:
            cpu_req = op.cpu_required
            mem_req = op.mem_required
            gpu_req = 0
            gpu_mem_req = 0
            base_resource_frac = 0.0

            if op.gpu_required:
                if not op.use_cuda():
                    raise ValueError(
                        f"Op[{op._name}] attempted to request GPU resources (gpu_required={op.gpu_required}), "
                        "but appears to lack GPU support. If you have verified this operator support GPU acceleration, "
                        'please explicitly set its property: `_accelerator = "cuda"`.'
                    )
                if not cuda_available:
                    raise ValueError(
                        f"Op[{op._name}] attempted to request GPU resources (gpu_required={op.gpu_required}), "
                        "but the gpu is unavailable. Please check whether your environment is installed correctly"
                        " and whether there is a gpu in the resource pool."
                    )

            # if it is a cuda operator, mem_required will be calculated as gpu memory;
            # if it is a cpu, it will be calculated as memory.

            auto_proc = False if op.num_proc else True

            # GPU operator calculations
            if op.use_cuda():
                gpu_req = op.gpu_required
                gpu_mem_req = op.mem_required
                if not gpu_req and not gpu_mem_req:
                    logger.warning(
                        f"The required cuda memory and gpu of Op[{op._name}] "
                        f"has not been specified. "
                        f"Please specify the `mem_required` field or `gpu_required` field in the "
                        f"config file. You can reference the `config_all.yaml` file."
                        f"Set the `gpu_required` to 1 now."
                    )
                    gpu_req = 1

                base_resource_frac = max(
                    cpu_req / total_cpu if cpu_req else 0,
                    gpu_req / total_gpu if gpu_req else 0,
                    gpu_mem_req / available_gpu_mem if gpu_mem_req else 0,
                )

                if not gpu_req:
                    gpu_req = math.ceil(base_resource_frac * total_gpu * 100) / 100
            # CPU operator calculations
            else:
                if cpu_req or mem_req:
                    base_resource_frac = max(
                        cpu_req / total_cpu if cpu_req else 0, mem_req / available_mem if mem_req else 0
                    )
                else:
                    logger.warning(
                        f"The required memory and cpu of Op[{op._name}] "
                        f"has not been specified. "
                        f"We recommend specifying the `mem_required` field or `cpu_required` field in the "
                        f"config file. You can reference the `config_all.yaml` file."
                    )
                    # Default to single CPU if no requirements specified
                    base_resource_frac = 1 / total_cpu

            resource_configs[op._name] = {
                "cpu_required": cpu_req,
                "gpu_required": gpu_req,
                "mem_required": mem_req,
                "gpu_mem_required": gpu_mem_req,
                "base_resource_frac": base_resource_frac,
                "num_proc": tuple(op.num_proc) if isinstance(op.num_proc, list) else op.num_proc,
                "auto_proc": auto_proc,
            }

        fixed_min_resources = 0
        fixed_max_resources = 0
        auto_resource_frac_map = {}
        for op_name, cfg in resource_configs.items():
            if cfg["auto_proc"]:
                auto_resource_frac_map[op_name] = cfg["base_resource_frac"]
            else:
                num_proc = cfg["num_proc"]
                min_proc = num_proc[0] if isinstance(num_proc, (tuple, list)) else num_proc
                max_proc = num_proc[1] if isinstance(num_proc, (tuple, list)) else num_proc
                fixed_min_resources += cfg["base_resource_frac"] * min_proc
                fixed_max_resources += cfg["base_resource_frac"] * max_proc

        # Validate resource availability
        total_auto_base_resource = sum(list(auto_resource_frac_map.values()))
        total_required_min = fixed_min_resources + total_auto_base_resource
        if total_required_min > 1:
            raise ValueError(
                f"Insufficient cluster resources: "
                f"At least {total_required_min:.2f}x the current resource is required. "
                f"Add resources or reduce operator requirements."
            )
        if len(auto_resource_frac_map) > 0:
            remaining_min_frac = 1 - fixed_max_resources
            remaining_max_frac = 1 - fixed_min_resources

            op_names, op_resources = [], []
            for k, v in auto_resource_frac_map.items():
                op_names.append(k)
                op_resources.append(v)
            best_combination, _, _ = find_optimal_concurrency(op_resources, remaining_min_frac)
            best_combination = dict(zip(op_names, best_combination))

            for op_name, cfg in resource_configs.items():
                if cfg["auto_proc"]:
                    min_proc = best_combination[op_name]
                    max_proc = int(max(1, remaining_max_frac / cfg["base_resource_frac"]))
                    cfg["num_proc"] = min_proc if min_proc == max_proc else (min_proc, max_proc)

        for op in operators:
            cfg = resource_configs[op._name]
            auto_proc, num_proc = cfg["auto_proc"], cfg["num_proc"]
            if op.use_cuda():
                op.cpu_required = cfg["cpu_required"]
                op.gpu_required = cfg["gpu_required"]
                op.num_proc = num_proc
            else:
                # * If ``fn`` is a function and ``concurrency`` is an  int ``n``, Ray Data
                # launches *at most* ``n`` concurrent tasks.
                op.cpu_required = cfg["cpu_required"]
                op.gpu_required = None
                # if concurrency left to None, the automatic concurrency of ray may be slightly higher, which could lead to OOM
                op.num_proc = num_proc[1] if (auto_proc and isinstance(num_proc, (tuple, list))) else num_proc
                # op.num_proc = None if auto_proc else num_proc

            logger.info(
                f"Op[{op._name}] will be executed with the following resources: "
                f"num_cpus: {op.cpu_required}, "
                f"num_gpus: {op.gpu_required}, "
                f"concurrency: {op.num_proc}, "
            )
        return operators

    def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]

        RayDataset.set_resource_for_ops(operators)

        for op in operators:
            self._run_single_op(op)
        return self

    def _run_single_op(self, op):
        if op._name in TAGGING_OPS.modules and Fields.meta not in self.data.columns():

            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.meta, [new_column_data])
                return new_table

            self.data = self.data.map_batches(
                process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
            )

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
                        num_cpus=op.cpu_required,
                        num_gpus=op.gpu_required,
                        concurrency=op.num_proc,
                        batch_format="pyarrow",
                    )
                else:
                    self.data = self.data.map_batches(
                        op.process,
                        batch_size=batch_size,
                        batch_format="pyarrow",
                        num_cpus=op.cpu_required,
                        concurrency=op.num_proc,
                    )
            elif isinstance(op, Filter):
                columns = self.data.columns()
                if Fields.stats not in columns:

                    def process_batch_arrow(table: pyarrow.Table):
                        new_column_data = [{} for _ in range(len(table))]
                        new_talbe = table.append_column(Fields.stats, [new_column_data])
                        return new_talbe

                    self.data = self.data.map_batches(
                        process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
                    )
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=None,
                        fn_constructor_kwargs=op_kwargs,
                        batch_size=batch_size,
                        num_cpus=op.cpu_required,
                        num_gpus=op.gpu_required,
                        concurrency=op.num_proc,
                        batch_format="pyarrow",
                    )
                else:
                    self.data = self.data.map_batches(
                        op.compute_stats,
                        batch_size=batch_size,
                        batch_format="pyarrow",
                        num_cpus=op.cpu_required,
                        concurrency=op.num_proc,
                    )
                if op.stats_export_path is not None:
                    self.data.write_json(op.stats_export_path, force_ascii=False)
                if op.is_batched_op():
                    # The core computation have been done in compute_stats,
                    # and the filter process only performs simple filtering.
                    # cpu and parallelism are not set here
                    self.data = self.data.map_batches(
                        partial(filter_batch, filter_func=op.process),
                        batch_format="pyarrow",
                        zero_copy_batch=True,
                        batch_size=DEFAULT_BATCH_SIZE,
                    )
                else:
                    self.data = self.data.filter(op.process)
            elif isinstance(op, Deduplicator):
                self.data = op.run(self.data)
            else:
                logger.error("Ray executor only support Filter, Mapper and Deduplicator OPs for now")
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
