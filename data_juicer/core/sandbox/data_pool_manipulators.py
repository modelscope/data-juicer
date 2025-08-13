import os
from datetime import datetime
from fractions import Fraction
from typing import get_args, get_origin

import jsonlines as jl
import numpy as np
from datasets import concatenate_datasets
from jsonargparse import dict_to_namespace
from loguru import logger

from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.data.dj_dataset import NestedDataset, NestedQueryDict
from data_juicer.core.data.schema import Schema
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import add_suffix_to_filename


def make_hashable(obj):
    if isinstance(obj, np.ndarray):
        return tuple(obj.flatten().tolist())  # 将数组转换为元组
    elif isinstance(obj, datetime):
        return obj.isoformat()  # 将时间转换为字符串
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, set)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, tuple):
        return tuple(make_hashable(e) for e in obj)
    else:
        return obj


def get_longest_common_prefix(list_of_strings):
    """Get the longest common prefix of the given list of strings."""
    if len(list_of_strings) == 0:
        return ""
    res = ""
    for chars in zip(*list_of_strings):
        if len(set(chars)) == 1:
            res += chars[0]
        else:
            break
    return res


def check_io_paths(input_paths, export_path):
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    existing_input_paths = []
    missing_paths = []
    for p in input_paths:
        if not os.path.exists(p):
            missing_paths.append(p)
        else:
            existing_input_paths.append(p)
    if len(missing_paths) > 0:
        logger.error(f'Input paths [{",".join(missing_paths)}] does not exist. Skipped!')
    if len(existing_input_paths) == 0:
        return None, None
    if export_path is None:
        raise ValueError("export_path is not specified.")
    os.makedirs(export_path, exist_ok=True)
    return existing_input_paths, export_path


def load_data_pool(ds_path) -> NestedDataset:
    """Load dataset. Can only return NestedDataset."""
    db = DatasetBuilder(dict_to_namespace({"dataset_path": ds_path, "text_keys": None}))
    ds = db.load_dataset()
    return ds


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
            - (optional) split_ratios. It's [1/3, 2/3] in default, Support fraction in string format.
            - (optional) split_num. It's not activated in default. Support specifying the number of samples in one data
                pool. Use split_num first if both of split_ratios and split_num are specified.
            - (optional) ignore_stats. It's False in default. Whether to split the data pool according to the stats
                ranking.
        Output: MxN data pools, where N is the number of types of analyzed stats and M means the number of split parts.
            If ignore_stats is True, N is 1 and the result data pools are stored in the export_path directly.
            They are named following the rule "<stats_key_name>/<original_name>_<part_idx>.jsonl"
        """
        # read inputs
        input_dataset_paths = self.data_pool_cfg.get("dataset_path", [])
        export_path = self.data_pool_cfg.get("export_path", None)
        split_ratios = self.data_pool_cfg.get("split_ratios", [1.0 / 3.0, 2.0 / 3.0])
        split_num = self.data_pool_cfg.get("split_num", None)
        ignore_stats = self.data_pool_cfg.get("ignore_stats", False)

        # check I/O paths
        existing_input_paths, export_path = check_io_paths(input_dataset_paths, export_path)
        # check split ratios, should be in (0, 1)
        split_ratios = [float(Fraction(r)) for r in split_ratios]  # make sure each ratio is a float
        if any([r <= 0 or r >= 1 for r in split_ratios]):
            raise ValueError("split_ratios should be in (0, 1).")

        # start to construct the data pools
        logger.info(f"Constructing data pools with split ratios {split_ratios}...")
        output_paths = []
        for ds_path in existing_input_paths:
            output_paths.extend(self._construct_data_pool(ds_path, export_path, split_ratios, split_num, ignore_stats))

        return output_paths

    def _construct_data_pool(self, ds_path, export_path, split_ratios, split_num, ignore_stats=False):
        logger.info(f"Constructing data pool for {ds_path}...")
        ds_basename = os.path.splitext(os.path.basename(ds_path))[0]
        ds = load_data_pool(ds_path)
        ds_schema = ds.schema()
        if not ignore_stats and Fields.stats not in ds_schema.columns:
            logger.warning(f"Dataset {ds_path} does not contain stats. Skipped!")
            return
        ds = ds.to_list()
        total_num = len(ds)
        if total_num == 0:
            logger.warning(f"Dataset {ds_path} is empty. Skipped!")
            return
        if split_num is not None:
            split_points = list(range(split_num, total_num, split_num))
        else:
            split_points = [int(total_num * r + 0.5) for r in split_ratios]
        split_points = [0] + split_points + [total_num]
        logger.info(f"Split points: {split_points}")

        # do not consider the stats information
        if ignore_stats:
            logger.info("Ignore stats and split the dataset with the current dataset state.")
            output_paths = []
            os.makedirs(export_path, exist_ok=True)
            for i in range(len(split_points) - 1):
                start_idx = split_points[i]
                end_idx = split_points[i + 1]
                part_ds = ds[start_idx:end_idx]
                curr_export_name = add_suffix_to_filename(ds_basename, f"_ignore_stats_{i}.jsonl")
                output_path = os.path.join(export_path, curr_export_name)
                with jl.open(output_path, "w") as writer:
                    writer.write_all(part_ds)
                output_paths.append(output_path)
            return output_paths

        stats_schema = ds_schema.column_types[Fields.stats]
        if not isinstance(stats_schema, Schema):
            logger.warning("Wrong structure of dataset stats. Skipped!")
            return
        stats_keys = stats_schema.columns
        if len(stats_keys) == 0:
            logger.warning(f"Dataset {ds_path} does not contain stats. Skipped!")
            return
        output_paths = []
        for stats_key in stats_keys:
            logger.info(f"Splitting data pools for stats key {stats_key}...")
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
                curr_export_name = add_suffix_to_filename(ds_basename, f"_{stats_key}_{i}.jsonl")
                output_path = os.path.join(stored_dir, curr_export_name)
                with jl.open(output_path, "w") as writer:
                    writer.write_all(part_ds)
                output_paths.append(output_path)
        return output_paths


class DataPoolCombination(BaseDataPoolManipulator):
    def run(self):
        """
        combine data pool from specified data pools

        Input:
            - N split data pools, which are already ordered by their ranks.
        Output: 2^N combined data pools including the original N data pools. Equals to N + C(N, 2) + ... + C(N, N). They
            are named following the rule "<longest_common_prefix>_top_<combined_ranks>_num_<num_samples>.jsonl"
        """
        # read inputs
        ordered_data_pool_paths = self.data_pool_cfg.get("dataset_path", [])
        export_path = self.data_pool_cfg.get("export_path", None)

        # check I/O paths
        existing_input_paths, output_path = check_io_paths(ordered_data_pool_paths, export_path)

        # start to combine these data pools
        logger.info("Combining data pools...")
        combined_hierarchies, dataset_records = self._combine_data_pools(existing_input_paths)
        # print hierarchies:
        # [
        #   [('0_1_2', num_samples)],
        #   [('0_1', num_samples), ('0_2', num_samples), ('1_2', num_samples)],
        #   [...],
        #   ...
        # ]
        print_msg = "\n"
        for pools in combined_hierarchies[::-1]:
            fmt_pools = [f"{cur_rank}: {len(pool)}" for cur_rank, pool in pools]
            print_msg += ";\t".join(fmt_pools) + "\n"
        logger.info(f"Combined hierarchies: {print_msg}")
        # export hierarchies
        longest_common_prefix = get_longest_common_prefix(
            [os.path.splitext(os.path.basename(p))[0] for p in existing_input_paths]
        )
        if longest_common_prefix == "":
            longest_common_prefix = "default_prefix"
        output_paths = []
        output_path_pattern = os.path.join(export_path, f"{longest_common_prefix}_top_%s_num_%d.jsonl")
        for pools in combined_hierarchies:
            for cur_rank, pool in pools:
                output_ds = [dataset_records[key] for key in pool]
                if len(output_ds) == 0:
                    continue
                output_path = output_path_pattern % (cur_rank, len(pool))
                with jl.open(output_path, "w") as writer:
                    writer.write_all(output_ds)
                output_paths.append(output_path)
        return output_paths

    def _combine_data_pools(self, ordered_data_pool_paths):
        curr_rank = 0
        dataset_records = {}
        hierarchies = []
        while True:
            # 1. read a new rank ds
            logger.info(f"Reading dataset of rank [{curr_rank}]...")
            ds = load_data_pool(ordered_data_pool_paths[curr_rank]).to_list()
            ds_dict = {make_hashable(s): s for s in ds}
            dataset_records.update(ds_dict)

            # 2. intersect new rank ds with each level in the hierarchies
            logger.info(f"Try to merge rank [{curr_rank}]...")
            new_items = [[(curr_rank, set(ds_dict.keys()))]]
            for pools in hierarchies:
                new_pools = []
                for this_rank, pool in pools:
                    new_rank = f"{this_rank}_{curr_rank}"
                    new_pool = pool.intersection(set(ds_dict.keys()))
                    new_pools.append((new_rank, new_pool))
                new_items.append(new_pools)

            # 3. count the top num
            top_num = len(new_items[-1][0][1])
            logger.info(f"After merging rank [{curr_rank}], there are [{top_num}] samples in the top level")

            # 4. merge to the hierarchies
            assert len(hierarchies) == len(new_items) - 1
            for i in range(len(hierarchies)):
                hierarchies[i].extend(new_items[i])
            hierarchies.append(new_items[-1])
            curr_rank += 1
            if curr_rank >= len(ordered_data_pool_paths):
                break
        return hierarchies, dataset_records


class DataPoolDuplication(BaseDataPoolManipulator):
    def run(self):
        """
        duplicate a data pool for specified times

        Input:
            - N specified data pools.
            - a list of duplicating times. E.g. [2, 4, 8]
            - whether to shuffle the duplicated dataset.
        Output: NxM new duplicated data pools, where M means the length of the times list. They are named following the
            rule "<original_name>_x<times>.jsonl"
        """
        # read inputs
        input_dataset_paths = self.data_pool_cfg.get("dataset_path", [])
        export_path = self.data_pool_cfg.get("export_path", None)
        dup_times = self.data_pool_cfg.get("duplicating_times", [])
        shuffle = self.data_pool_cfg.get("shuffle", False)
        seed = self.data_pool_cfg.get("shuffle_seed", 42)

        # check I/O paths
        existing_input_paths, export_path = check_io_paths(input_dataset_paths, export_path)
        # check duplicating times, should be int >= 1
        if any([not isinstance(r, int) or r <= 0 for r in dup_times]):
            raise ValueError("duplicating_times should be integers >= 1.")

        logger.info(f"Duplicating data pools for {dup_times} times with shuffle [{shuffle}]...")
        output_paths = []
        for input_dataset in input_dataset_paths:
            output_paths.extend(self._duplicate_dataset(input_dataset, export_path, dup_times, shuffle, seed))
        return output_paths

    def _duplicate_dataset(self, dataset_path, export_path, dup_times, shuffle=False, seed=42):
        logger.info(f"Duplicating dataset for {dataset_path}...")
        ds_basename = os.path.splitext(os.path.basename(dataset_path))[0]
        ds = load_data_pool(dataset_path)
        output_paths = []
        for t in dup_times:
            res_ds = concatenate_datasets([ds] * t)
            if shuffle:
                res_ds = res_ds.shuffle(seed=seed).flatten_indices()
            output_path = os.path.join(export_path, f"{ds_basename}_x{t}.jsonl")
            res_ds.to_json(output_path)
            output_paths.append(output_path)
        return output_paths


class DataPoolRanking(BaseDataPoolManipulator):
    def run(self):
        """
        rank data pools according to specified evaluation metrics.

        Input:
            - N specified data pools
            - The evaluated metrics of these N data pools in dict with data paths as keys.
            - (optional) Keys in the metrics to rank the data pools. Support '.' operator to get a nested key. Use the
                whole metric obj in default.
            - (optional) whether to sort in descending. It's True in default
            - (optional) a number N that only return the top-N data pool paths.
        Output: A ordered list of data pool paths according to their evaluated metrics.
        """
        input_dataset_paths = self.data_pool_cfg.get("dataset_path", [])
        metrics = self.data_pool_cfg.get("metrics", [])
        ranking_keys = self.data_pool_cfg.get("ranking_keys", [])
        descending = self.data_pool_cfg.get("descending", True)
        top_n = self.data_pool_cfg.get("top_n", None)

        # check input paths and metrics
        if (
            not isinstance(input_dataset_paths, list)
            or not isinstance(metrics, list)
            or len(input_dataset_paths) != len(metrics)
        ):
            raise ValueError("dataset_path and metrics should be lists of the same length.")

        # check ranking keys
        sampled_metrics = NestedQueryDict(metrics[0])
        metrics = [NestedQueryDict(m) for m in metrics]
        existing_keys = []
        missing_keys = []
        for key in ranking_keys:
            if sampled_metrics[key] is not None:
                existing_keys.append(key)
            else:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            logger.error(f'Ranking keys [{",".join(missing_keys)}] does not exist. Skipped!')

        # key func that extracts key info to rank
        logger.info(
            f'Ranking data pools on keys [{",".join(existing_keys)}] with '
            f'[{"descending" if descending else "ascending"}]...'
        )

        def _key_func(zipped_metric):
            return tuple(zipped_metric[1][k] for k in existing_keys)

        # default key func that uses the whole metric obj to rank
        def _default_key_func(zipped_metric):
            return zipped_metric[1]

        if len(existing_keys) <= 0:
            selected_key_func = _default_key_func
        else:
            selected_key_func = _key_func

        # sort by metrics
        ranked_paths = [
            zipped[0] for zipped in sorted(zip(input_dataset_paths, metrics), key=selected_key_func, reverse=descending)
        ]
        if top_n is not None:
            logger.info(f"Select only top-{top_n} data pools...")
            ranked_paths = ranked_paths[:top_n]
        return ranked_paths


class DataPoolDownsampling(BaseDataPoolManipulator):
    def run(self):
        """
        Randomly downsample data pools to specified scale.

        Input:
            - N specified data pools.
            - (optional) the target number of samples. It's decided by the smallest data pool in default.
            - (optional) seed for randomness.
        Output: N downsampled data pools. They are named following the rule "<original_name>_<num_sample>.jsonl"
        """
        # read inputs
        input_dataset_paths = self.data_pool_cfg.get("dataset_path", [])
        export_path = self.data_pool_cfg.get("export_path", None)
        target_num = self.data_pool_cfg.get("target_num_samples", None)
        seed = self.data_pool_cfg.get("shuffle_seed", 42)

        # check I/O paths
        existing_input_paths, export_path = check_io_paths(input_dataset_paths, export_path)

        # load all datasets
        all_datasets = [load_data_pool(path) for path in existing_input_paths]
        all_lengths = [len(ds) for ds in all_datasets]
        if target_num is None:
            target_num = min(all_lengths)

        logger.info(f"Downsampling data pools to {target_num} samples...")
        all_datasets = [ds.shuffle(seed=seed).take(min(target_num, len(ds))) for ds in all_datasets]
        output_paths = []
        for ds, path in zip(all_datasets, existing_input_paths):
            ds_basename = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(export_path, f"{ds_basename}_{target_num}.jsonl")
            ds.to_json(output_path)
            output_paths.append(output_path)
        return output_paths


class DataPoolMerging(BaseDataPoolManipulator):
    def run(self):
        """
        merge data pools into one dataset or data pool.

        Input:
            - N split data pools.
        Output: 1 merged dataset/data pool, which is named following the rule "<longest_common_prefix>_merged.jsonl"
        """
        # read inputs
        ordered_data_pool_paths = self.data_pool_cfg.get("dataset_path", [])
        export_path = self.data_pool_cfg.get("export_path", None)

        # check I/O paths
        existing_input_paths, output_path = check_io_paths(ordered_data_pool_paths, export_path)

        # start to combine these data pools
        logger.info("Merging data pools...")
        # try to get the longest_common_prefix
        longest_common_prefix = get_longest_common_prefix(
            [os.path.splitext(os.path.basename(p))[0] for p in existing_input_paths]
        )
        if longest_common_prefix == "":
            longest_common_prefix = "default_prefix"
        output_path = os.path.join(export_path, f"{longest_common_prefix}_merged.jsonl")
        data_pools = [load_data_pool(path) for path in existing_input_paths]
        merged_dataset = concatenate_datasets(data_pools)
        merged_dataset.to_json(output_path, force_ascii=False)
        return output_path


class DataPoolCartesianJoin(BaseDataPoolManipulator):
    def run(self):
        """
        join two sets of data pools with Cartesian Join.

        Example: Given two sets of data pools M and N, where M = {DP(A, B, C), DP(E, F), DP(G, H, I, J)} and
            N = {DP(1), DP(2, 3)}. After this hook, they are Cartesian joined to:
            {
                DP(A1, B1, C1),
                DP(A2, A3, B2, B3, C2, C3),
                DP(E1, F1),
                DP(E2, E3, F2, F3),
                DP(G1, H1, I1, J1),
                DP(G2, G3, H2, H3, I2, I3, J2, J3),
            }

        Input:
            - M data pools.
            - N data pools.
        Output: M x N joined data pools MN, where MN(i, j) = M(i) x N(j).
            They are named following the rule "<longest_common_prefix>_cartesian_join_{i}_{j}.jsonl"
        """
        # read inputs
        first_data_pool_paths = self.data_pool_cfg.get("dataset_path_1", [])
        second_data_pool_paths = self.data_pool_cfg.get("dataset_path_2", [])
        export_path = self.data_pool_cfg.get("export_path", "")

        # check I/O paths
        first_existing_input_paths, output_path = check_io_paths(first_data_pool_paths, export_path)
        second_existing_input_paths, output_path = check_io_paths(second_data_pool_paths, output_path)
        first_num_data_pools = len(first_existing_input_paths)
        second_num_data_pools = len(second_existing_input_paths)

        # start to combine these data pools
        logger.info(
            f"Cartesian join two sets of data pools with "
            f"{first_num_data_pools} and {second_num_data_pools} data pools..."
        )
        # try to get the longest_common_prefix
        first_longest_common_prefix = get_longest_common_prefix(
            [os.path.splitext(os.path.basename(p))[0] for p in first_existing_input_paths]
        )
        second_longest_common_prefix = get_longest_common_prefix(
            [os.path.splitext(os.path.basename(p))[0] for p in second_existing_input_paths]
        )
        longest_common_prefix = f"{first_longest_common_prefix}_{second_longest_common_prefix}"
        if longest_common_prefix == "_":
            longest_common_prefix = "default_prefix"
        output_path_pattern = os.path.join(export_path, f"{longest_common_prefix}_cartesian_join_%d_%d.jsonl")
        output_paths = []
        for i, first_path in enumerate(first_existing_input_paths):
            for j, second_path in enumerate(second_existing_input_paths):
                output_path = output_path_pattern % (i, j)
                first_dataset = load_data_pool(first_path)
                second_dataset = load_data_pool(second_path)
                joined_dataset = self._cartesian_join_two_dataset(first_dataset, second_dataset)
                joined_dataset.to_json(output_path, force_ascii=False)
                output_paths.append(output_path)
        return output_paths

    def _cartesian_join_two_dataset(self, first_dataset: NestedDataset, second_dataset: NestedDataset):
        len1 = len(first_dataset)
        len2 = len(second_dataset)
        first_repeated = concatenate_datasets([NestedDataset.from_list([d] * len2) for d in first_dataset])
        second_repeated = concatenate_datasets([second_dataset] * len1)
        return concatenate_datasets([first_repeated, second_repeated], axis=1)
