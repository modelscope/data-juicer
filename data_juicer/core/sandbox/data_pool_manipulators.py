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
            They are named following the rule "<original_name>_<part_idx>.jsonl"
        """
        raise NotImplementedError


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
