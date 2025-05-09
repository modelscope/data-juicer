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
