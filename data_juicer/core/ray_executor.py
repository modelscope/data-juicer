import time

from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core.ray_data import RayDataset
from data_juicer.ops import load_ops
from data_juicer.utils.availability_utils import AvailabilityChecking

with AvailabilityChecking(['ray'], requires_type='dist'):
    import ray


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

        # init ray
        logger.info('Initing Ray ...')
        ray.init(self.cfg.ray_address)

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
            dataset = RayDataset.read_jsonl(self.cfg.dataset_path, self.cfg)
        # 2. extract processes
        logger.info('Preparing process operators...')
        ops = load_ops(self.cfg.process, self.cfg.op_fusion)

        # 3. data process
        logger.info('Processing data...')
        tstart = time.time()
        dataset.process(ops)

        # 4. data export
        logger.info('Exporting dataset to disk...')
        dataset.write_json(self.cfg.export_path, force_ascii=False)

        tend = time.time()
        logger.info(f'All Ops are done in {tend - tstart:.3f}s.')

        return dataset
