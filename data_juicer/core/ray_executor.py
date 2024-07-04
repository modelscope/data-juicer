import os
import time
from functools import partial

import pandas as pd
import pyarrow as pa
from loguru import logger

from data_juicer import cuda_device_count, use_cuda
from data_juicer.config import init_configs
from data_juicer.ops import Filter, Mapper, load_ops
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.process_utils import calculate_np

with AvailabilityChecking(['ray'], requires_type='dist'):
    import ray
    import ray.data as rd
    from ray.data import ActorPoolStrategy

from data_juicer.ops.base_op import OPERATORS


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


def ray_batch_mapper_wrapper(samples, fn):
    samples = samples.to_pandas()
    res = fn(samples)
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame(res)
    return pa.Table.from_pandas(res)


class RayExecutor:
    """
    Executor based on Ray.

    Run Data-Juicer data processing in a distributed cluster.

        1. Support Filter, Mapper and Exact Deduplicator operators for now.
        2. Only support loading `.json` files.
        3. Advanced functions such as checkpoint, tracer are not supported.

    """

    def __init__(self, cfg=None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        self.cfg = init_configs() if cfg is None else cfg

        self.work_dir = self.cfg.work_dir

        self.ops = None
        # init ray
        logger.info('Initing Ray ...')
        ray.init(self.cfg.ray_address)
        self.process_list = self.cfg.process

    def get_num_gpus(self, op, op_proc):
        if not use_cuda() or not op._accelerator == 'cuda':
            return 0
        proc_per_gpu = op_proc / cuda_device_count()
        return 1.0 / proc_per_gpu

    def run_op(self, op, op_cfg, dataset):
        op_name, op_args = list(op_cfg.items())[0]
        op_cls = OPERATORS.modules[op_name]
        op_proc = calculate_np(self.cfg.np, op, op_name)
        num_gpus = self.get_num_gpus(op, op_proc)
        use_actor = op.use_actor() or num_gpus
        try:
            if isinstance(op, Mapper):
                if op.is_batched_op():
                    if use_actor:
                        dataset = dataset.map_batches(
                            op_cls,
                            compute=ActorPoolStrategy(),
                            concurrency=op_proc,
                            fn_constructor_kwargs=op_args,
                            batch_format='pyarrow',
                            num_gpus=num_gpus,
                            batch_size=1)
                        # The batch size here is same as in data.py
                    else:
                        dataset = dataset.map_batches(partial(
                            ray_batch_mapper_wrapper, fn=op.process),
                                                      batch_format='pyarrow',
                                                      num_gpus=num_gpus,
                                                      batch_size=1)
                        # The batch size here is same as in data.py
                else:
                    if use_actor:
                        dataset = dataset.map(op_cls,
                                              compute=ActorPoolStrategy(),
                                              concurrency=op_proc,
                                              fn_constructor_kwargs=op_args,
                                              num_gpus=num_gpus)
                    else:
                        dataset = dataset.map(op.process, num_gpus=num_gpus)

            elif isinstance(op, Filter):
                if use_actor:
                    dataset = dataset.map(op_cls,
                                          compute=ActorPoolStrategy(),
                                          concurrency=op_proc,
                                          fn_constructor_kwargs=op_args,
                                          num_gpus=num_gpus)
                else:
                    dataset = dataset.map(op.compute_stats, num_gpus=num_gpus)
                if op.stats_export_path is not None:
                    dataset.write_json(op.stats_export_path, force_ascii=False)
                dataset = dataset.filter(op.process)
            else:
                logger.error(
                    'Ray executor only support Filter and Mapper OPs for '
                    'now')
                raise NotImplementedError

            return dataset
        except:  # noqa: E722
            logger.error(f'An error occurred during Op [{op_name}].')
            import traceback
            traceback.print_exc()
            exit(1)

    def run(self, load_data_np=None):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        # 1. load data
        logger.info('Loading dataset with Ray...')
        dataset = rd.read_json(self.cfg.dataset_path)

        # convert all the path in dataset to absolute path
        dataset = set_dataset_to_absolute_path(dataset, self.cfg.dataset_path,
                                               self.cfg)
        logger.info('Dataset columns:', dataset.columns())
        # 2. extract processes
        logger.info('Preparing process operators...')
        self.process_list, self.ops = load_ops(self.cfg.process,
                                               self.cfg.op_fusion)

        # 3. data process
        # - If tracer is open, trace each op after it's processed
        # - If checkpoint is open, clean the cache files after each process
        if Fields.stats not in dataset.columns(fetch_if_missing=False):
            logger.info(f'columns {dataset.columns(fetch_if_missing=False)}')

            def process_batch_arrow(table: pa.Table) -> pa.Table:
                new_column_data = [{} for _ in range(len(table))]
                new_talbe = table.append_column(Fields.stats,
                                                [new_column_data])
                return new_talbe

            dataset = dataset.map_batches(process_batch_arrow,
                                          batch_format='pyarrow')

        logger.info('Processing data...')
        tstart = time.time()
        for op_cfg, op in zip(self.process_list, self.ops):
            dataset = self.run_op(op, op_cfg, dataset)

        # 4. data export
        logger.info('Exporting dataset to disk...')
        dataset.write_json(self.cfg.export_path, force_ascii=False)
        tend = time.time()
        logger.info(f'All Ops are done in {"%.3f" % (tend - tstart)}(s).')
        return dataset
