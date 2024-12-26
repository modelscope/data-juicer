from __future__ import annotations

import os
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import pyarrow
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.data import DJDataset
from data_juicer.ops import Deduplicator, Filter, Mapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.process_utils import calculate_np

rd = LazyLoader('rd', 'ray.data')
ds = LazyLoader('ds', 'ray.data.datasource')


def get_abs_path(path, dataset_dir):
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
                samples[key][idx] = [
                    get_abs_path(item, dataset_dir) for item in paths
                ]
    return pyarrow.Table.from_pydict(samples)


# TODO: check path for nestdataset
def set_dataset_to_absolute_path(dataset, dataset_path, cfg):
    """
    Set all the path in input data to absolute path.
    Checks dataset_dir and project_dir for valid paths.
    """
    path_keys = []
    columns = dataset.columns()
    for key in [cfg.video_key, cfg.image_key, cfg.audio_key]:
        if key in columns:
            path_keys.append(key)
    if len(path_keys) > 0:
        dataset_dir = os.path.dirname(dataset_path)
        dataset = dataset.map_batches(partial(convert_to_absolute_paths,
                                              dataset_dir=dataset_dir,
                                              path_keys=path_keys),
                                      batch_format='pyarrow',
                                      zero_copy_batch=True)
    return dataset


def preprocess_dataset(dataset: rd.Dataset, dataset_path, cfg) -> rd.Dataset:
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

    def __init__(self,
                 dataset: rd.Dataset,
                 dataset_path: str = None,
                 cfg=None) -> None:
        self.data = preprocess_dataset(dataset, dataset_path, cfg)
        self.num_proc = None
        if cfg:
            self.num_proc = cfg.np

    def process(self,
                operators,
                *,
                exporter=None,
                checkpointer=None,
                tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]
        for op in operators:
            self._run_single_op(op)
        return self

    def _run_single_op(self, op):
        op_proc = calculate_np(op._name, op.mem_required, op.cpu_required,
                               self.num_proc, op.use_cuda())
        num_gpus = get_num_gpus(op, op_proc)
        try:
            batch_size = getattr(op, 'batch_size',
                                 1) if op.is_batched_op() else 1
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
                        batch_format='pyarrow')
                else:
                    self.data = self.data.map_batches(op.process,
                                                      batch_size=batch_size,
                                                      batch_format='pyarrow',
                                                      num_gpus=num_gpus)
            elif isinstance(op, Filter):
                columns = self.data.columns()
                if Fields.stats not in columns:

                    def process_batch_arrow(table: pyarrow.Table):
                        new_column_data = [{} for _ in range(len(table))]
                        new_talbe = table.append_column(
                            Fields.stats, [new_column_data])
                        return new_talbe

                    self.data = self.data.map_batches(process_batch_arrow,
                                                      batch_format='pyarrow')
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
                        batch_format='pyarrow')
                else:
                    self.data = self.data.map_batches(op.compute_stats,
                                                      batch_size=batch_size,
                                                      batch_format='pyarrow',
                                                      num_gpus=num_gpus)
                if op.stats_export_path is not None:
                    self.data.write_json(op.stats_export_path,
                                         force_ascii=False)
                if op.is_batched_op():
                    self.data = self.data.map_batches(partial(
                        filter_batch, filter_func=op.process),
                                                      batch_format='pyarrow',
                                                      batch_size=batch_size,
                                                      num_gpus=num_gpus,
                                                      zero_copy_batch=True)
                else:
                    self.data = self.data.filter(op.process)
            elif isinstance(op, Deduplicator):
                self.data = op.run(self.data)
            else:
                logger.error(
                    'Ray executor only support Filter and Mapper OPs for now')
                raise NotImplementedError
        except:  # noqa: E722
            logger.error(f'An error occurred during Op [{op._name}].')
            import traceback
            traceback.print_exc()
            exit(1)

    @classmethod
    def read_json(cls, paths: Union[str, List[str]]) -> RayDataset:
        # Note: a temp solution for reading json stream
        # TODO: replace with ray.data.read_json_stream once it is available
        import pyarrow.json as js
        try:
            js.open_json
            return read_json_stream(paths)
        except AttributeError:
            return rd.read_json(paths)


class JSONStreamDatasource(ds.JSONDatasource):
    """
    A temp Datasource for reading json stream.

    Note:

        Depends on a customized `pyarrow` with `open_json` method.
    """

    def _read_stream(self, f: 'pyarrow.NativeFile', path: str):
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
            raise ValueError(f'Failed to read JSON file: {path}.') from e


def read_json_stream(
    paths: Union[str, List[str]],
    *,
    filesystem: Optional['pyarrow.fs.FileSystem'] = None,
    parallelism: int = -1,
    ray_remote_args: Dict[str, Any] = None,
    arrow_open_stream_args: Optional[Dict[str, Any]] = None,
    meta_provider=None,
    partition_filter=None,
    partitioning=ds.partitioning.Partitioning('hive'),
    include_paths: bool = False,
    ignore_missing_paths: bool = False,
    shuffle: Union[Literal['files'], None] = None,
    file_extensions: Optional[List[str]] = ['json', 'jsonl'],
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
    **arrow_json_args,
) -> rd.Dataset:
    if meta_provider is None:
        meta_provider = ds.file_meta_provider.DefaultFileMetadataProvider()

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
    return rd.read_datasource(
        datasource,
        parallelism=parallelism,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )
