import os
from time import time

from loguru import logger

from data_juicer.config import init_configs
from data_juicer.format.load import load_formatter
from data_juicer.ops import (OPERATORS, Deduplicator, Filter, Mapper, Selector,
                             load_ops)
from data_juicer.utils import cache_utils
from data_juicer.utils.ckpt_utils import CheckpointManager
from data_juicer.utils.constant import Fields

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

        self.ops = None

        # only enable it when using cache
        if self.cfg.use_cache:
            logger.info(f'Using cache compression method: '
                        f'[{self.cfg.cache_compress}]')
            cache_utils.CACHE_COMPRESS = self.cfg.cache_compress

        # setup formatter
        logger.info('Setting up data formatter...')
        self.formatter = load_formatter(self.cfg.dataset_path,
                                        self.cfg.text_keys, self.cfg.suffixes,
                                        self.cfg.add_suffix)

        # whether to use checkpoint mechanism. If it's true, Executor will
        # check if there are existing checkpoints first and try to load the
        # checkpoints. If the checkpoints are loaded successfully, ops that
        # have been processed will be skipped.
        self.process_list = self.cfg.process
        if self.cfg.use_checkpoint:
            logger.info('Preparing checkpoint manager...')
            self.ckpt_dir = os.path.join(self.work_dir, 'ckpt')
            self.ckpt_manager = CheckpointManager(self.ckpt_dir,
                                                  self.process_list,
                                                  self.cfg.np)
            if self.ckpt_manager.ckpt_available:
                logger.info('Found existed dataset checkpoint.')
                self.process_list = self.ckpt_manager.get_left_process_list()
        self.cfg.process = self.process_list

        # prepare exporter and check export path suffix
        logger.info('Preparing exporter...')
        self.exporter = Exporter(self.cfg.export_path,
                                 self.cfg.export_shard_size,
                                 self.cfg.export_in_parallel,
                                 self.cfg.np)

        # setup tracer
        self.open_tracer = self.cfg.open_tracer
        if self.open_tracer:
            logger.info('Preparing tracer...')
            self.tracer = Tracer(self.work_dir, show_num=self.cfg.trace_num)
            self.op_list_to_trace = self.cfg.op_list_to_trace
            if len(self.cfg.op_list_to_trace) == 0:
                logger.info('Trace for all ops.')
                self.op_list_to_trace = set(OPERATORS.modules.keys())

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
            dataset = self.formatter.load_dataset(load_data_np)

        # 2. extract processes
        logger.info('Preparing process operators...')
        self.process_list, self.ops = load_ops(self.cfg.process,
                                               self.cfg.op_fusion)

        # 3. data process
        # - If tracer is open, trace each op after it's processed
        # - If checkpoint is open, clean the cache files after each process
        logger.info('Processing data...')
        start = time()
        tstart = start
        for op_cfg, op in zip(self.process_list, self.ops):
            op_name, op_args = list(op_cfg.items())[0]
            prev = dataset  # record last dataset
            try:
                if isinstance(op, Mapper):
                    tmp = dataset.map(function=op.process,
                                      num_proc=self.cfg.np,
                                      desc=op_name + '_process')
                    if self.open_tracer and \
                            op_name in self.op_list_to_trace:
                        if op.is_batched_op():
                            self.tracer.trace_batch_mapper(
                                op_name,
                                dataset,
                                tmp,
                                op.text_key)
                        else:
                            self.tracer.trace_mapper(op_name,
                                                     dataset,
                                                     tmp,
                                                     op.text_key)
                elif isinstance(op, Filter):
                    if Fields.stats not in dataset.features:
                        # TODO:
                        # this is a temp solution,
                        # only add stats when calling filter op
                        dataset = dataset.add_column(name=Fields.stats,
                                                     column=[{}] *
                                                     dataset.num_rows)
                        if self.cfg.use_checkpoint:
                            prev = dataset
                    dataset = dataset.map(op.compute_stats,
                                          num_proc=self.cfg.np,
                                          desc=op_name + '_compute_stats')
                    if self.cfg.use_checkpoint:
                        prev = dataset
                    tmp = dataset.filter(op.process,
                                         num_proc=self.cfg.np,
                                         desc=op_name + '_process')
                    if self.open_tracer and op_name in self.op_list_to_trace:
                        self.tracer.trace_filter(op_name, dataset, tmp)
                elif isinstance(op, Selector):
                    tmp = op.process(dataset)
                    if self.open_tracer and op_name in self.op_list_to_trace:
                        self.tracer.trace_filter(op_name, dataset, tmp)
                elif isinstance(op, Deduplicator):
                    dataset = dataset.map(op.compute_hash,
                                          num_proc=self.cfg.np,
                                          desc=op_name + '_compute_hash')
                    if self.cfg.use_checkpoint:
                        prev = dataset
                    tmp, dup_pairs = op.process(
                        dataset, self.tracer.show_num if self.open_tracer
                        and op_name in self.op_list_to_trace else 0)
                    if self.open_tracer and op_name in self.op_list_to_trace:
                        self.tracer.trace_deduplicator(op_name, dup_pairs)
                else:
                    raise NotImplementedError
                dataset = tmp
            except:  # noqa: E722
                logger.error(f'An error occurred during Op [{op_name}].')
                import traceback
                traceback.print_exc()
                if self.cfg.use_checkpoint:
                    logger.info('Writing checkpoint of dataset processed by '
                                'last op...')
                    prev.cleanup_cache_files()
                    self.ckpt_manager.save_ckpt(prev)
                exit(1)

            # clean up cache files and record processed ops
            if self.cfg.use_checkpoint:
                self.ckpt_manager.record(op_name, op_args)

            end = time()
            logger.info(f'Op [{op_name}] Done in {"%.3f" % (end - start)}(s). '
                        f'Left {len(dataset)} samples.')
            start = end
        tend = time()
        logger.info(f'All Ops are done in {"%.3f" % (tend - tstart)}(s).')

        # 4. data export
        logger.info('Exporting dataset to disk...')
        try:
            self.exporter.export(dataset)
        except:  # noqa: E722
            logger.error('An error occurred during exporting the processed '
                         'dataset.')
            import traceback
            traceback.print_exc()
            if self.cfg.use_checkpoint:
                logger.info('Writing checkpoint of dataset processed by '
                            'last op...')
                dataset.cleanup_cache_files()
                self.ckpt_manager.save_ckpt(dataset)
        # compress the last dataset after exporting
        if self.cfg.use_cache and self.cfg.cache_compress:
            from data_juicer.utils.compress import compress
            compress(dataset)
        return dataset
