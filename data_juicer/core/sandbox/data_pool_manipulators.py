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
            - split_ratios. (optional, [1/3, 2/3] in default)
            - target number of samples. (optional)
        Output: MxN data pools, where N is the number of types of analyzed stats and M means the number of split parts.
            They are named following the rule "<original_name>_<part_idx>_<num_samples>.jsonl"
        """
        raise NotImplementedError


class DataPoolCombination(BaseDataPoolManipulator):

    def run(self):
        """
        combine data pool from specified data pools
        """
        raise NotImplementedError


class DataPoolDuplication(BaseDataPoolManipulator):

    def run(self):
        """
        duplicate a data pool for specified times
        """
        raise NotImplementedError


class DataPoolRanking(BaseDataPoolManipulator):

    def run(self):
        """
        rank data pools according to specified evaluation metrics.
        """
        raise NotImplementedError


class DataPoolDownsampling(BaseDataPoolManipulator):

    def run(self):
        """
        downsample data pools to specified scale.
        """
        raise NotImplementedError
