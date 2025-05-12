import os
from typing import get_args, get_origin

import jsonlines as jl
import numpy as np
from jsonargparse import dict_to_namespace
from loguru import logger

from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.data.schema import Schema
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import add_suffix_to_filename


class BaseDataPoolManipulator(object):

    def __init__(self, data_pool_cfg: dict):
        self.data_pool_cfg = data_pool_cfg

    def run(self):
        """
        Manipulations for data pools.
        """
        raise NotImplementedError


class DataPoolConstruction(BaseDataPoolManipulator):

    def run(self):
        """
        construct data pool from specified analyzed data source

        Input:
            - an analyzed dataset.
            - an output path.
            - (optional) split_ratios. It's [1/3, 2/3] in default
        Output: MxN data pools, where N is the number of types of analyzed stats and M means the number of split parts.
            They are named following the rule "<stats_key_name>/<original_name>_<part_idx>.jsonl"
        """
        # read inputs
        input_dataset_path = self.data_pool_cfg.get('dataset_path', [])
        export_path = self.data_pool_cfg.get('export_path', None)
        split_ratios = self.data_pool_cfg.get('split_ratios',
                                              [1.0 / 3.0, 2.0 / 3.0])

        # check I/O paths
        if isinstance(input_dataset_path, str):
            input_dataset_path = [input_dataset_path]
        existing_input_paths = []
        missing_paths = []
        for p in input_dataset_path:
            if not os.path.exists(p):
                missing_paths.append(p)
            else:
                existing_input_paths.append(p)
        if len(missing_paths) > 0:
            logger.error(
                f'Input dataset paths [{",".join(missing_paths)}] does not exist. Skipped!'
            )
        if len(existing_input_paths) == 0:
            return None
        if export_path is None:
            raise ValueError('export_path is not specified.')
        os.makedirs(export_path, exist_ok=True)

        # start to construct the data pools
        for ds_path in existing_input_paths:
            self._construct_data_pool(ds_path, export_path, split_ratios)

        return export_path

    def _construct_data_pool(self, ds_path, export_path, split_ratios):
        logger.info(f'Constructing data pool for {ds_path}...')
        ds_basename = os.path.splitext(os.path.basename(ds_path))[0]
        db = DatasetBuilder(dict_to_namespace({'dataset_path': ds_path}))
        ds = db.load_dataset()
        ds_schema = ds.schema()
        if Fields.stats not in ds_schema.columns:
            logger.warning(
                f'Dataset {ds_path} does not contain stats. Skipped!')
            return
        ds = ds.to_list()
        total_num = len(ds)
        if total_num == 0:
            logger.warning(f'Dataset {ds_path} is empty. Skipped!')
            return
        split_points = [int(total_num * r + 0.5) for r in split_ratios]
        split_points = [0] + split_points + [total_num]
        logger.info(f'Split points: {split_points}')

        stats_schema = ds_schema.column_types[Fields.stats]
        if not isinstance(stats_schema, Schema):
            logger.warning('Wrong structure of dataset stats. Skipped!')
            return
        stats_keys = stats_schema.columns
        if len(stats_keys) == 0:
            logger.warning(
                f'Dataset {ds_path} does not contain stats. Skipped!')
            return
        for stats_key in stats_keys:
            logger.info(f'Splitting data pools for stats key {stats_key}...')
            # 1. sort by this key
            # stats_type can only be numbers, string, or list of numbers or strings.
            stats_type = stats_schema.column_types[stats_key]
            origin_type = get_origin(stats_type)
            arg_type = get_args(stats_type)
            if len(arg_type) == 0:
                arg_type = None
            else:
                arg_type = arg_type[0]
            if origin_type is list and arg_type is not None and arg_type is not str:
                # list of numbers
                def key_func(lst):
                    return np.mean(lst) if len(lst) > 0 else 0
            elif origin_type is None and stats_type is not str:
                # numbers
                def key_func(v):
                    return v
            else:
                # other types, just skip
                continue
            # sort by this stats key
            ds.sort(key=lambda s: key_func(s[Fields.stats][stats_key]))

            # 2. split by split_points
            stored_dir = os.path.join(export_path, stats_key)
            os.makedirs(stored_dir, exist_ok=True)
            for i in range(len(split_points) - 1):
                start_idx = split_points[i]
                end_idx = split_points[i + 1]
                part_ds = ds[start_idx:end_idx]
                curr_export_name = add_suffix_to_filename(
                    ds_basename, f'_{i}.jsonl')
                with jl.open(os.path.join(stored_dir, curr_export_name),
                             'w') as writer:
                    writer.write_all(part_ds)


class DataPoolCombination(BaseDataPoolManipulator):

    def run(self):
        """
        combine data pool from specified data pools

        Input:
            - N split data pools, with their ranks.
        Output: 2^N combined data pools including the original N data pools. Equals to N + C(N, 2) + ... + C(N, N). They
            are named following the rule "<most_common_prefix>_top_<combined_ranks>.jsonl"
        """
        raise NotImplementedError


class DataPoolDuplication(BaseDataPoolManipulator):

    def run(self):
        """
        duplicate a data pool for specified times

        Input:
            - N specified data pools.
            - a list of duplicating times. E.g. [2, 4, 8]
        Output: NxM new duplicated data pools, where M means the length of the times list. They are named following the
            rule "<original_name>_x<times>.jsonl"
        """
        raise NotImplementedError


class DataPoolRanking(BaseDataPoolManipulator):

    def run(self):
        """
        rank data pools according to specified evaluation metrics.

        Input:
            - N specified data pools with their evaluated metrics.
            - (optional) Some ranking methods or rules. Ranked in descending order in default.
        Output: A ordered list of data pool paths according to their evaluated metrics.
        """
        raise NotImplementedError


class DataPoolDownsampling(BaseDataPoolManipulator):

    def run(self):
        """
        downsample data pools to specified scale.

        Input:
            - N specified data pools.
            - (optional) the target number of samples. It's decided by the smallest data pool in default.
        Output: N downsampled data pools. They are named following the rule "<original_name>_<num_sample>.jsonl"
        """
        raise NotImplementedError
