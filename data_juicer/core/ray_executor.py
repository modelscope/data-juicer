import os
import shutil
import time

from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core.ray_data import RayDataset
from data_juicer.ops import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.lazy_loader import LazyLoader

from .adapter import Adapter

ray = LazyLoader('ray', 'ray')
rd = LazyLoader('rd', 'ray.data')


class TempDirManager:

    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def __enter__(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.tmp_dir):
            logger.info(f'Removing tmp dir {self.tmp_dir} ...')
            shutil.rmtree(self.tmp_dir)


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

        self.adapter = Adapter(self.cfg)

        # init ray
        logger.info('Initing Ray ...')
        ray.init(self.cfg.ray_address)
        self.tmp_dir = os.path.join(self.work_dir, '.tmp',
                                    ray.get_runtime_context().get_job_id())

    def run(self, load_data_np=None):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        # 1. load data
        logger.info('Loading dataset with Ray...')

        if self.cfg.get('generated_dataset_config', None):
            generated_dataset_config = self.cfg.generated_dataset_config
            assert isinstance(generated_dataset_config,
                              dict) and 'type' in generated_dataset_config
            args = generated_dataset_config.copy()
            obj_name = args.pop('type')
            from data_juicer.format.formatter import FORMATTERS
            dataset = FORMATTERS.modules[obj_name](**args).load_dataset()
        else:
            dataset = RayDataset.read_json(self.cfg.dataset_path)

        # convert all the path in dataset to absolute path
        dataset = RayDataset(dataset, self.cfg.dataset_path, self.cfg)
        # 2. extract processes
        logger.info('Preparing process operators...')
        ops = load_ops(self.cfg.process)

        if self.cfg.op_fusion:
            probe_res = None
            if self.cfg.fusion_strategy == 'probe':
                logger.info('Probe the OP speed for OP reordering...')
                probe_res, _ = self.adapter.probe_small_batch(dataset, ops)

            logger.info(f'Start OP fusion and reordering with strategy '
                        f'[{self.cfg.fusion_strategy}]...')
            ops = fuse_operators(ops, probe_res)

        with TempDirManager(self.tmp_dir):
            # 3. data process
            logger.info('Processing data...')
            tstart = time.time()
            dataset.process(ops)

            # 4. data export
            logger.info('Exporting dataset to disk...')
            dataset.data.write_json(self.cfg.export_path, force_ascii=False)
            tend = time.time()
            logger.info(f'All Ops are done in {tend - tstart:.3f}s.')
        return dataset
