from abc import ABC, abstractmethod
from typing import Union

from data_juicer.core.data import DJDataset, RayDataset

# based on executor type and data source type, use different
# data load strategy to product corresponding datasets
# DJDataset, RayDataset, DaskDataset, etc


class DataLoadStrategyRegistry:

    def __init__(self):
        self._registry = {}

    def register(self, key: tuple, strategy):
        """Register a strategy for a specific tuple key."""
        if key in self._registry:
            raise ValueError(f'Strategy for key {key} is already registered.')
        self._registry[key] = strategy

    def get_strategy(self, key: tuple):
        """Retrieve the strategy for a specific tuple key."""
        if key not in self._registry:
            raise ValueError(f'No strategy registered for key {key}.')
        return self._registry[key]

    def register_decorator(self, key: tuple):
        """Decorator for registering a strategy with a specific tuple key."""

        def decorator(func):
            self.register(key, func)
            return func  # Return the original function

        return decorator


DATALOAD_STRATEGY_REGISTRY = DataLoadStrategyRegistry()


class DataLoadStrategyFactory:

    @classmethod
    def create_dataload_strategy(cls, executor_type, dataset_type,
                                 dataset_source):
        DATALOAD_STRATEGY_REGISTRY.get_strategy(
            (executor_type, dataset_type, dataset_source))


class DataLoadStrategy(ABC):

    @abstractmethod
    def load_data(self) -> Union[DJDataset, RayDataset]:
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(('ray', 'ondisk', 'json'))
class RayOndiskJsonDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(
    ('ray', 'remote', 'huggingface'))
class RayHuggingfaceDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(('local', 'ondisk', 'Json'))
class LocalOndiskJsonDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(('local', 'ondisk', 'Parquet'))
class LocalOndiskParquetDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(
    ('local', 'remote', 'huggingface'))
class LocalHuggingfaceDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(
    ('local', 'remote', 'modelscope'))
class LocalModelScopeDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(('local', 'remote', 'arxiv'))
class LocalArxivDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(('local', 'remote', 'wiki'))
class LocalWikiDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass


@DATALOAD_STRATEGY_REGISTRY.register_decorator(
    ('local', 'remote', 'commoncrawl'))
class LocalCommonCrawlDataLoadStrategy(DataLoadStrategy):

    def load_data(self):
        pass
