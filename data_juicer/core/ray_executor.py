import time

from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core.ray_data import RayDataset
from data_juicer.ops import load_ops
from data_juicer.utils.availability_utils import AvailabilityChecking

with AvailabilityChecking(['ray'], requires_type='dist'):
    import ray
    import ray.data as rd


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
        dataset = RayDataset(dataset, self.cfg.dataset_path, self.cfg)
        # 2. extract processes
        logger.info('Preparing process operators...')
        self.process_list, self.ops = load_ops(self.cfg.process,
                                               self.cfg.op_fusion)

        # 3. data process
        logger.info('Processing data...')
        tstart = time.time()
        dataset.process(self.ops)
        tend = time.time()
        logger.info(f'All Ops are done in {tend - tstart:.3f}s.')

        # 4. data export
        logger.info('Exporting dataset to disk...')
        dataset.data.write_json(self.cfg.export_path, force_ascii=False)
        return dataset
