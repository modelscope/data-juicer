import math
import os
import subprocess
from time import time

import psutil
import torch
from loguru import logger

from data_juicer import cuda_device_count, use_cuda
from data_juicer.config import init_configs
from data_juicer.format.load import load_formatter
from data_juicer.ops import (OPERATORS, Deduplicator, Filter, Mapper, Selector,
                             load_ops)
from data_juicer.utils import cache_utils
from data_juicer.utils.ckpt_utils import CheckpointManager
from data_juicer.utils.constant import Fields

from .data import add_same_content_to_new_column
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

    def get_min_cuda_memory(self):
        # get cuda memory info using "nvidia-smi" command
        min_cuda_memory = torch.cuda.get_device_properties(
            0).total_memory / 1024**2
        nvidia_smi_output = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8')
        for line in nvidia_smi_output.strip().split('\n'):
            free_memory = int(line)
            min_cuda_memory = min(min_cuda_memory, free_memory)
        return min_cuda_memory

    def calculate_np(self, op, op_name):
        if use_cuda() and op._accelerator == 'cuda':
            cuda_mem_available = self.get_min_cuda_memory() / 1024
            op_proc = min(
                self.cfg.np,
                math.floor(cuda_mem_available / (op.mem_required + 0.1)) *
                cuda_device_count())
            if op_proc < 1.0:
                logger.warning(
                    f'The required cuda memory:{op.mem_required}GB might '
                    f'be more than the available cuda memory:'
                    f'{cuda_mem_available}GB.'
                    f'This Op [{op_name}] might '
                    f'require more resource to run.')
            op_proc = max(op_proc, 1)
            return op_proc
        else:
            op_proc = self.cfg.np
            cpu_available = psutil.cpu_count()
            mem_available = psutil.virtual_memory().available
            mem_available = mem_available / 1024**3
            op_proc = min(op_proc, math.floor(cpu_available / op.cpu_required))
            op_proc = min(op_proc,
                          math.floor(mem_available / (op.mem_required + 0.1)))
            if op_proc < 1.0:
                logger.warning(
                    f'The required CPU number:{op.cpu_required} '
                    f'and memory:{op.mem_required}GB might '
                    f'be more than the available CPU:{cpu_available} '
                    f'and memory :{mem_available}GB.'
                    f'This Op [{op_name}] might '
                    f'require more resource to run.')
            op_proc = max(op_proc, 1)
            return op_proc

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
            with_rank = use_cuda() and op._accelerator == 'cuda'
            if op.spec_numprocs != 0:
                op_proc = op.spec_numprocs
                logger.info(f'Op [{op_name}] running with sepcified '
                            f'number of procs:{op.spec_numprocs}')
            else:
                op_proc = self.calculate_np(op, op_name)
            try:
                if isinstance(op, Mapper):
                    tmp = dataset.map(function=op.process,
                                      num_proc=op_proc,
                                      with_rank=with_rank,
                                      desc=op_name + '_process')
                    if self.open_tracer and \
                            op_name in self.op_list_to_trace:
                        if op.is_batched_op():
                            self.tracer.trace_batch_mapper(
                                op_name, dataset, tmp, op.text_key)
                        else:
                            self.tracer.trace_mapper(op_name, dataset, tmp,
                                                     op.text_key)
                elif isinstance(op, Filter):
                    if Fields.stats not in dataset.features:
                        # only add stats when calling filter op
                        dataset = dataset.map(
                            add_same_content_to_new_column,
                            fn_kwargs={
                                'new_column_name': Fields.stats,
                                'initial_value': {}
                            },
                            num_proc=self.cfg.np,
                            desc='Adding new column for stats')
                        if self.cfg.use_checkpoint:
                            prev = dataset
                    dataset = dataset.map(op.compute_stats,
                                          num_proc=op_proc,
                                          with_rank=with_rank,
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
                                          num_proc=op_proc,
                                          with_rank=with_rank,
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
