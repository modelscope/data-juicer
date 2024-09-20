import os

from loguru import logger

from data_juicer.analysis import ColumnWiseAnalysis, OverallAnalysis
from data_juicer.config import init_configs
from data_juicer.format import load_formatter
from data_juicer.ops import Filter, load_ops
from data_juicer.utils import cache_utils

from .exporter import Exporter


class Analyzer:
    """
    This Analyzer class is used to analyze a specific dataset.

    It will compute stats for all filter ops in the config file, apply
    multiple analysis (e.g. OverallAnalysis, ColumnWiseAnalysis, etc.)
    on these stats, and generate the analysis results (stats tables,
    distribution figures, etc.) to help users understand the input
    dataset better.
    """

    def __init__(self, cfg=None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        self.cfg = init_configs() if cfg is None else cfg

        self.work_dir = self.cfg.work_dir

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

        # prepare exporter and check export path suffix
        # NOTICE: no need to export dataset texts for analyzer
        # (export_ds=False). Instead, only need to export stats
        # (export_stats=True).
        logger.info('Preparing exporter...')
        self.exporter = Exporter(
            self.cfg.export_path,
            self.cfg.export_shard_size,
            self.cfg.export_in_parallel,
            self.cfg.np,
            export_ds=self.cfg.export_original_dataset,
            keep_stats_in_res_ds=self.cfg.export_original_dataset,
            export_stats=True)

        # parsed_res
        self.overall_result = None
        self.overall_single_plot_path = None
        self.analysis_path = os.path.join(self.cfg.work_dir, 'analysis')

    def run(self, load_data_np=None, skip_export=False):
        """
        Running the dataset analysis pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :param skip_export: whether export the results into disk
        :return: analyzed dataset.
        """
        # 1. format data
        logger.info('Loading dataset from data formatter...')
        if load_data_np is None:
            load_data_np = self.cfg.np
        dataset = self.formatter.load_dataset(load_data_np, self.cfg)

        # extract processes
        logger.info('Preparing process operators...')
        ops = load_ops(self.cfg.process, self.cfg.op_fusion)

        # 2. stats precompute only for filter ops
        logger.info('Computing the stats of dataset...')
        stats_collected = False
        for op in ops:
            if isinstance(op, Filter):
                original_process = op.process
                op.process = None
                dataset = dataset.process(op, work_dir=self.work_dir)
                op.process = original_process
                stats_collected = True
        if not stats_collected:
            logger.warning('No stats collected. Please add some Filter ops to '
                           'the process list in configs.')
            return dataset

        # 3. data export
        logger.info('Exporting dataset to disk...')
        self.exporter.export(dataset)
        if self.cfg.use_cache and self.cfg.cache_compress:
            from data_juicer.utils.compress import compress
            compress(dataset)

        # 4. analysis and output result to the export path
        # 4.1. Only consider fields in Fields.stats
        # 4.2. For string fields, only consider its histogram
        # 4.3. For numeric fields, consider its histogram and box
        # 4.4. Otherwise, DO NOT analyze

        logger.info('Applying overall analysis on stats...')
        overall_analysis = OverallAnalysis(dataset, self.analysis_path)
        self.overall_result = overall_analysis.analyze(
            percentiles=self.cfg.percentiles,
            num_proc=self.cfg.np,
            skip_export=skip_export)

        logger.info(f'The overall analysis results are: {self.overall_result}')

        logger.info('Applying column-wise analysis on stats...')
        column_wise_analysis = ColumnWiseAnalysis(
            dataset,
            self.analysis_path,
            overall_result=self.overall_result,
            save_stats_in_one_file=self.cfg.save_stats_in_one_file,
        )
        column_wise_analysis.analyze(skip_export=skip_export)

        return dataset
