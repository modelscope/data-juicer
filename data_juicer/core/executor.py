import math
import os
from time import time

import numpy as np
from datasets import load_dataset
from loguru import logger

from data_juicer import use_cuda
from data_juicer.config import init_configs
from data_juicer.core.data import Dataset
from data_juicer.format.load import load_formatter
from data_juicer.format.mixture_formatter import MixtureFormatter
from data_juicer.ops import (OPERATORS, Deduplicator, Filter, Mapper, Selector,
                             load_ops)
from data_juicer.utils import cache_utils
from data_juicer.utils.ckpt_utils import CheckpointManager
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.file_utils import add_suffix_to_filename
from data_juicer.utils.process_utils import calculate_np

from ..ops.selector.frequency_specified_field_selector import \
    FrequencySpecifiedFieldSelector
from ..ops.selector.topk_specified_field_selector import \
    TopkSpecifiedFieldSelector
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

    def export_dataset(self, exporter, dataset):
        """
            export the dataset
        """
        logger.info('Exporting dataset to disk...')
        try:
            exporter.export(dataset)
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

    def divide_by_percentiles(self,
                              load_data_np=None,
                              analyser_results=None,
                              **kwargs):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :param analyser_results: the overall analysis results from analyser
        :return: processed dataset.
        """
        # 1. load dataset with stats
        dataset = load_dataset('json',
                               data_files=self.cfg.export_path)['train']

        # 2. get the first stats key which has the mean val as the dimension
        #    for partitioning
        op_name_to_stats_key = StatsKeys.get_access_log(dj_cfg=self.cfg)
        mean_series = analyser_results[analyser_results.index == 'mean']
        stats_key_to_mean = mean_series.iloc[0, :].to_dict()
        stats_key_for_partition = None
        for process in self.cfg.process:
            op_name, args = list(process.items())[0]
            if op_name not in op_name_to_stats_key:
                # skip the op such as `clean_email_mapper`
                continue
            for stats_key in op_name_to_stats_key[op_name]:
                if stats_key in stats_key_to_mean and not math.isnan(
                        stats_key_to_mean[stats_key]):
                    stats_key_for_partition = stats_key
                    break

        # 3. get all the partition points in analyser_results
        def convert_string_to_float(s):
            new_s = s.replace(',', '').replace('%', '').replace('$', '')
            try:
                value = float(new_s)
                return value / 100 if '%' in s else value
            except ValueError:
                return None

        all_partition_points = {}
        for index, val in analyser_results[stats_key_for_partition].items():
            index_float = convert_string_to_float(index)
            if index_float is not None:
                all_partition_points[index_float] = val

        # 4. shift the partition_points according percentiles
        partition_vals = []
        for percentile in self.cfg.percentiles:
            all_percentiles = list(all_partition_points.keys())
            closest_percentile = min(all_percentiles,
                                     key=lambda p: abs(p - percentile))
            partition_vals.append(all_partition_points[closest_percentile])
        partition_vals.sort()

        # 5. partision dataset
        # TODO: reduce following as a selector op
        partitions = []
        stats_vals = [
            np.asarray(
                dataset[i][Fields.stats][stats_key_for_partition]).mean()
            for i in range(len(dataset))
        ]
        sort_index = np.argsort(np.array(stats_vals))
        each_num = len(stats_vals) // (len(partition_vals) + 1)
        base = 0
        for _ in partition_vals:
            partitions.append(dataset.select(sort_index[base:base + each_num]))
            base = base + each_num
        partitions.append(dataset.select(sort_index[base:]))

        # 6. data export
        logger.info('Exporting partition dataset to disk...')
        for i, dataset in enumerate(partitions):
            # fix the data num for each partition if necessary
            if self.cfg.fix_data_num is not None:
                if len(dataset) < self.cfg.fix_data_num:
                    logger.warning(f'dataset size {len(dataset)}'
                                   f' < fix num {self.cfg.fix_data_num}')
                dataset = MixtureFormatter.random_sample(
                    dataset, sample_number=self.cfg.fix_data_num)
                suffix = f'_part{i}_num{self.cfg.fix_data_num}'
            else:
                suffix = f'_part{i}'
            exporter = Exporter(
                add_suffix_to_filename(self.cfg.export_path, suffix),
                self.cfg.export_shard_size,
                self.cfg.export_in_parallel,
                self.cfg.np,
                keep_stats_in_res_ds=self.cfg.keep_stats_in_res_ds,
                keep_hashes_in_res_ds=self.cfg.keep_hashes_in_res_ds)
            self.export_dataset(exporter, dataset)

        return dataset

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
            dataset = MixtureFormatter.random_sample(dataset, sample_ratio)
        elif sample_algo == 'frequency_specified_field_selector':
            dj_op = FrequencySpecifiedFieldSelector(**kwargs)
            dataset = dj_op.process(dataset)
        elif sample_algo == 'topk_specified_field_selector':
            dj_op = TopkSpecifiedFieldSelector(**kwargs)
            dataset = dj_op.process(dataset)
        else:
            raise ValueError(f'Unsupported sample_algo: {sample_algo}')

        self.export_dataset(self.exporter, dataset)

        return dataset

    def run(self, load_data_np=None, **kwargs):
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
            if op.num_proc != 0:
                op_proc = op.num_proc
                logger.info(f'Op [{op_name}] running with sepcified '
                            f'number of procs:{op.num_proc}')
            else:
                op_proc = calculate_np(self.cfg.np, op, op_name)
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
                    if op.stats_export_path is not None:
                        self.exporter.export_compute_stats(
                            dataset, op.stats_export_path)
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
        self.export_dataset(self.exporter, dataset)
        return dataset
