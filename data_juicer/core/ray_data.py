import os

import pyarrow as pa
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.data import DJDataset
from data_juicer.ops import Filter, Mapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.process_utils import calculate_np

rd = LazyLoader('rd', 'ray.data')


def is_valid_path(item, dataset_dir):
    full_path = os.path.abspath(os.path.join(dataset_dir, item))
    return os.path.exists(full_path)


def convert_to_absolute_paths(dict_with_paths, dataset_dir, path_keys):
    for key in path_keys:
        if key not in dict_with_paths:
            continue
        if isinstance(dict_with_paths[key], list):
            dict_with_paths[key] = [
                os.path.abspath(os.path.join(dataset_dir, item))
                if isinstance(item, str) and is_valid_path(dataset_dir, item)
                else item for item in dict_with_paths[key]
            ]
        elif isinstance(dict_with_paths[key], str):
            dict_with_paths[key] = os.path.abspath(
                os.path.join(dataset_dir,
                             dict_with_paths[key])) if is_valid_path(
                                 dict_with_paths[key],
                                 dataset_dir) else dict_with_paths[key]
    return dict_with_paths


# TODO: check path for nestdataset
def set_dataset_to_absolute_path(dataset, dataset_path, cfg):
    """
    Set all the path in input data to absolute path.
    Checks dataset_dir and project_dir for valid paths.
    """
    if not (cfg.video_key in dataset.columns() or cfg.image_key
            in dataset.columns() or cfg.audio_key in dataset.columns()):
        return dataset
    dataset_dir = os.path.dirname(dataset_path)
    dataset = dataset.map(lambda item: convert_to_absolute_paths(
        item, dataset_dir, [cfg.video_key, cfg.image_key, cfg.audio_key]))
    logger.info(f"transfer {dataset.count()} sample's paths")
    return dataset


def preprocess_dataset(dataset: rd.Dataset, dataset_path, cfg) -> rd.Dataset:
    if dataset_path:
        dataset = set_dataset_to_absolute_path(dataset, dataset_path, cfg)
    columns = dataset.columns()
    if Fields.stats not in columns:
        logger.info(f'columns {columns}')

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
