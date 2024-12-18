import os
from functools import partial

import pyarrow as pa
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.data import DJDataset
from data_juicer.ops import Filter, Mapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.process_utils import calculate_np

rd = LazyLoader('rd', 'ray.data')


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
    return pa.Table.from_pydict(samples)


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
    columns = dataset.columns()
    if dataset_path:
        dataset = set_dataset_to_absolute_path(dataset, dataset_path, cfg)
    if Fields.stats not in columns:

        def process_batch_arrow(table: pa.Table) -> pa.Table:
            new_column_data = [{} for _ in range(len(table))]
            new_talbe = table.append_column(Fields.stats, [new_column_data])
            return new_talbe

        dataset = dataset.map_batches(process_batch_arrow,
                                      batch_format='pyarrow')
    return dataset


def get_num_gpus(op, op_proc):
    if not op.use_cuda():
        return 0
    proc_per_gpu = op_proc / cuda_device_count()
    return 1.0 / proc_per_gpu


def filter_batch(batch, filter_func):
    mask = pa.array(filter_func(batch.to_pydict()))
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
                self.data = self.data.map_batches(op.process,
                                                  batch_size=batch_size,
                                                  batch_format='pyarrow',
                                                  num_gpus=num_gpus)
            elif isinstance(op, Filter):
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
            else:
                logger.error(
                    'Ray executor only support Filter and Mapper OPs for now')
                raise NotImplementedError
        except:  # noqa: E722
            logger.error(f'An error occurred during Op [{op._name}].')
            import traceback
            traceback.print_exc()
            exit(1)
