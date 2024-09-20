import os
from time import time

from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core.data import Dataset
from data_juicer.format.load import load_formatter
from data_juicer.format.mixture_formatter import MixtureFormatter
from data_juicer.ops import OPERATORS, load_ops
from data_juicer.utils import cache_utils
from data_juicer.utils.ckpt_utils import CheckpointManager

from ..ops.selector.frequency_specified_field_selector import \
    FrequencySpecifiedFieldSelector
from ..ops.selector.topk_specified_field_selector import \
    TopkSpecifiedFieldSelector
from .exporter import Exporter
from .tracer import Tracer


class Executor:
    """
    This Executor class is used to process a specific dataset.

    It will load the dataset and unify the format, then apply all the
    ops in the config file in order and generate a processed dataset.
    """

    def __init__(self, cfg=None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        self.cfg = init_configs() if cfg is None else cfg

        self.work_dir = self.cfg.work_dir

        self.tracer = None
        self.ckpt_manager = None

        # only enable it when using cache
        if self.cfg.use_cache:
            logger.info(f'Using cache compression method: '
                        f'[{self.cfg.cache_compress}]')
            cache_utils.CACHE_COMPRESS = self.cfg.cache_compress

        # setup formatter
        logger.info('Setting up data formatter...')
        self.formatter = load_formatter(
            dataset_path=self.cfg.dataset_path,
            generated_dataset_config=self.cfg.generated_dataset_config,
            text_keys=self.cfg.text_keys,
            suffixes=self.cfg.suffixes,
            add_suffix=self.cfg.add_suffix)

        # whether to use checkpoint mechanism. If it's true, Executor will
        # check if there are existing checkpoints first and try to load the
        # checkpoints. If the checkpoints are loaded successfully, ops that
        # have been processed will be skipped.
        if self.cfg.use_checkpoint:
            logger.info('Preparing checkpoint manager...')
            self.ckpt_dir = os.path.join(self.work_dir, 'ckpt')
            self.ckpt_manager = CheckpointManager(self.ckpt_dir,
                                                  self.cfg.process,
                                                  self.cfg.np)
            if self.ckpt_manager.ckpt_available:
                logger.info('Found existed dataset checkpoint.')
                self.cfg.process = self.ckpt_manager.get_left_process_list()

        # prepare exporter and check export path suffix
        logger.info('Preparing exporter...')
        self.exporter = Exporter(
            self.cfg.export_path,
            self.cfg.export_shard_size,
            self.cfg.export_in_parallel,
            self.cfg.np,
            keep_stats_in_res_ds=self.cfg.keep_stats_in_res_ds,
            keep_hashes_in_res_ds=self.cfg.keep_hashes_in_res_ds)

        # setup tracer
        self.open_tracer = self.cfg.open_tracer
        if self.open_tracer:
            logger.info('Preparing tracer...')
            self.tracer = Tracer(self.work_dir, show_num=self.cfg.trace_num)
            self.op_list_to_trace = self.cfg.op_list_to_trace
            if len(self.cfg.op_list_to_trace) == 0:
                logger.info('Trace for all ops.')
                self.op_list_to_trace = set(OPERATORS.modules.keys())

    def sample_data(self,
                    dataset_to_sample: Dataset = None,
                    load_data_np=None,
                    sample_ratio: float = 1.0,
                    sample_algo: str = 'uniform',
                    **kwargs):
        """
        Sample a subset from the given dataset.

        :param dataset_to_sample: Dataset to sample from. If None, will use
            the formatter linked by the executor. Default is None.
        :param load_data_np: number of workers when loading the dataset.
        :param sample_ratio: The ratio of the sample size to the original
            dataset size. Default is 1.0 (no sampling).
        :param sample_algo: Sampling algorithm to use. Options are "uniform",
            "frequency_specified_field_selector", or
            "topk_specified_field_selector".
            Default is "uniform".
        :return: A sampled Dataset.
        """
        # Determine the dataset to sample from
        if dataset_to_sample is not None:
            dataset = dataset_to_sample
        elif self.cfg.use_checkpoint and self.ckpt_manager.ckpt_available:
            logger.info('Loading dataset from checkpoint...')
            dataset = self.ckpt_manager.load_ckpt()
        elif hasattr(self, 'formatter'):
            logger.info('Loading dataset from data formatter...')
            if load_data_np is None:
                load_data_np = self.cfg.np
            dataset = self.formatter.load_dataset(load_data_np, self.cfg)
        else:
            raise ValueError('No dataset available to sample from.')

        # Perform sampling based on the specified algorithm
        if sample_algo == 'uniform':
            return MixtureFormatter.random_sample(dataset, sample_ratio)
        elif sample_algo == 'frequency_specified_field_selector':
            dj_op = FrequencySpecifiedFieldSelector(**kwargs)
            return dj_op.process(dataset)
        elif sample_algo == 'topk_specified_field_selector':
            dj_op = TopkSpecifiedFieldSelector(**kwargs)
            return dj_op.process(dataset)
        else:
            raise ValueError(f'Unsupported sample_algo: {sample_algo}')

    def run(self, load_data_np=None):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        # 1. format data
        if self.cfg.use_checkpoint and self.ckpt_manager.ckpt_available:
            logger.info('Loading dataset from checkpoint...')
            dataset = self.ckpt_manager.load_ckpt()
        else:
            logger.info('Loading dataset from data formatter...')
            if load_data_np is None:
                load_data_np = self.cfg.np
            dataset = self.formatter.load_dataset(load_data_np, self.cfg)

        # 2. extract processes
        logger.info('Preparing process operators...')
        ops = load_ops(self.cfg.process, self.cfg.op_fusion)

        # 3. data process
        # - If tracer is open, trace each op after it's processed
        # - If checkpoint is open, clean the cache files after each process
        logger.info('Processing data...')
        tstart = time()
        dataset = dataset.process(ops,
                                  work_dir=self.work_dir,
                                  exporter=self.exporter,
                                  checkpointer=self.ckpt_manager,
                                  tracer=self.tracer)
        tend = time()
        logger.info(f'All OPs are done in {tend - tstart:.3f}s.')

        # 4. data export
        logger.info('Exporting dataset to disk...')
        self.exporter.export(dataset)
        # compress the last dataset after exporting
        if self.cfg.use_cache and self.cfg.cache_compress:
            from data_juicer.utils.compress import compress
            compress(dataset)
        return dataset
