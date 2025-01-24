import fnmatch
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, Optional, Type, Union

import datasets

from data_juicer.core.data import DJDataset, RayDataset
from data_juicer.core.data.config_validator import ConfigValidator
from data_juicer.download.downloader import validate_snapshot_format
from data_juicer.format.formatter import unify_format
from data_juicer.format.load import load_formatter
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader('ray', 'ray')
rd = LazyLoader('rd', 'ray.data')

# based on executor type and data source type, use different
# data load strategy to product corresponding datasets
# DJDataset, RayDataset, DaskDataset, etc


@dataclass(frozen=True)
class StrategyKey:
    """
    Immutable key for strategy registration with wildcard support
    """
    executor_type: str
    data_type: str
    data_source: str

    def matches(self, other: 'StrategyKey') -> bool:
        """
        Check if this key matches another key with wildcard support

        Supports Unix-style wildcards:
        - '*' matches any string
        - '?' matches any single character
        - '[seq]' matches any character in seq
        - '[!seq]' matches any character not in seq
        """
        return (fnmatch.fnmatch(other.executor_type, self.executor_type)
                and fnmatch.fnmatch(other.data_type, self.data_type)
                and fnmatch.fnmatch(other.data_source, self.data_source))


class DataLoadStrategy(ABC, ConfigValidator):
    """
    abstract class for data load strategy
    """

    def __init__(self, ds_config: Dict, cfg: Namespace):
        self.validate_config(ds_config)
        self.ds_config = ds_config
        self.cfg = cfg
        self.weight = ds_config.get('weight', 1.0)  # default weight is 1.0

    @abstractmethod
    def load_data(self, **kwargs) -> Union[DJDataset, RayDataset]:
        pass


class DataLoadStrategyRegistry:
    """
    Flexible strategy registry with wildcard matching
    """
    _strategies: Dict[StrategyKey, Type[DataLoadStrategy]] = {}

    @classmethod
    def get_strategy_class(
            cls, executor_type: str, data_type: str,
            data_source: str) -> Optional[Type[DataLoadStrategy]]:
        """
        Retrieve the most specific matching strategy

        Matching priority:
        1. Exact match
        2. Wildcard matches from most specific to most general
        """
        # default to wildcard if not provided
        executor_type = executor_type or '*'
        data_type = data_type or '*'
        data_source = data_source or '*'

        # Create the lookup key
        lookup_key = StrategyKey(executor_type, data_type, data_source)

        # First, check for exact match
        exact_match = cls._strategies.get(lookup_key)
        if exact_match:
            return exact_match

        # Find all matching wildcard strategies
        matching_strategies = []
        for registered_key, strategy in cls._strategies.items():
            if registered_key.matches(lookup_key):
                matching_strategies.append((registered_key, strategy))

        # Sort matching strategies by specificity (fewer wildcards first)
        if matching_strategies:

            def specificity_score(key: StrategyKey) -> int:
                """
                Calculate specificity score (lower is more specific)
                Exact match: 0
                One wildcard: 1
                Two wildcards: 2
                All wildcards: 3
                """
                return sum(1 for part in
                           [key.executor_type, key.data_type, key.data_source]
                           if part == '*')

            matching_strategies.sort(key=lambda x: specificity_score(x[0]))
            return matching_strategies[0][1]

        # No matching strategy found
        return None

    @classmethod
    def register(cls, executor_type: str, data_type: str, data_source: str):
        """
        Decorator for registering data load strategies with wildcard support

        :param executor_type: Type of executor (e.g., 'local', 'ray')
        :param data_type: Type of data (e.g., 'ondisk', 'remote')
        :param data_source: Specific data source (e.g., 'arxiv', 's3')
        :return: Decorator function
        """

        def decorator(strategy_class: Type[DataLoadStrategy]):
            """
            Register the strategy class for the given key

            :param strategy_class: Strategy class to register
            :return: Original strategy class
            """
            key = StrategyKey(executor_type, data_type, data_source)
            cls._strategies[key] = strategy_class
            return strategy_class

        return decorator


class RayDataLoadStrategy(DataLoadStrategy):
    """
    abstract class for data load strategy for RayExecutor
    """

    @abstractmethod
    def load_data(self, **kwargs) -> RayDataset:
        pass


class LocalDataLoadStrategy(DataLoadStrategy):
    """
    abstract class for data load strategy for LocalExecutor
    """

    @abstractmethod
    def load_data(self, **kwargs) -> DJDataset:
        pass


# TODO dask support
# class DaskDataLoadStrategy(DataLoadStrategy):
#     @abstractmethod
#     def load_data(self) -> Union[DaskDataset]:
#         pass

# TODO nemo support
# class NemoDataLoadStrategy(DataLoadStrategy):
#     @abstractmethod
#     def load_data(self) -> Union[NemoDataset]:
#         pass


@DataLoadStrategyRegistry.register('ray', 'ondisk', '*')
class RayOndiskJsonDataLoadStrategy(RayDataLoadStrategy):

    # TODO ray defaults to json

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': str
        },
        'custom_validators': {}
    }

    def load_data(self, **kwargs):
        dataset = rd.read_json(self.ds_config['path'])
        return RayDataset(dataset,
                          dataset_path=self.ds_config['path'],
                          cfg=self.cfg)


@DataLoadStrategyRegistry.register('ray', 'remote', 'huggingface')
class RayHuggingfaceDataLoadStrategy(RayDataLoadStrategy):

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': str
        },
        'custom_validators': {}
    }

    def load_data(self, **kwargs):
        raise NotImplementedError(
            'Huggingface data load strategy is not implemented')


@DataLoadStrategyRegistry.register('local', 'ondisk', '*')
class LocalOndiskDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for on disk data for LocalExecutor
    rely on AutoFormatter for actual data loading
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': str
        },
        'custom_validators': {}
    }

    def load_data(self, **kwargs):
        print(f'kwards: {kwargs}')
        # use proper formatter to load data
        formatter = load_formatter(dataset_path=self.ds_config['path'],
                                   suffixes=self.cfg.suffixes,
                                   text_keys=self.cfg.text_keys,
                                   add_suffix=self.cfg.add_suffix,
                                   **kwargs)
        # TODO more sophiscated localformatter routing
        return formatter.load_dataset()


@DataLoadStrategyRegistry.register('local', 'remote', 'huggingface')
class LocalHuggingfaceDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for Huggingface dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'optional_fields': ['split', 'limit', 'name'],
        'field_types': {
            'path': str
        },
        'custom_validators': {}
    }

    def load_data(self, **kwargs):
        num_proc = kwargs.pop('num_proc', 1)
        ds = datasets.load_dataset(self.ds_config['path'],
                                   split=self.ds_config.split,
                                   name=self.ds_config.name,
                                   limit=self.ds_config.limit,
                                   num_proc=num_proc,
                                   **kwargs)
        ds = unify_format(ds, text_keys=self.text_keys, num_proc=num_proc)


@DataLoadStrategyRegistry.register('local', 'remote', 'modelscope')
class LocalModelScopeDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for ModelScope dataset for LocalExecutor
    """

    def load_data(self):
        raise NotImplementedError(
            'ModelScope data load strategy is not implemented')


@DataLoadStrategyRegistry.register('local', 'remote', 'arxiv')
class LocalArxivDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for arxiv dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': (str)  # has to be a string
        },
        'custom_validators': {}
    }

    def load_data(self):
        raise NotImplementedError(
            'Arxiv data load strategy is not implemented')


@DataLoadStrategyRegistry.register('local', 'remote', 'wiki')
class LocalWikiDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for wiki dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': str
        },
        'custom_validators': {}
    }

    def load_data(self):
        raise NotImplementedError('Wiki data load strategy is not implemented')


@DataLoadStrategyRegistry.register('local', 'remote', 'commoncrawl')
class LocalCommonCrawlDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for commoncrawl dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['start_snapshot', 'end_snapshot'],
        'optional_fields': ['aws', 'url_limit'],
        'field_types': {
            'start_snapshot': str,
            'end_snapshot': str
        },
        'custom_validators': {
            'start_snashot': validate_snapshot_format,
            'end_snapshot': validate_snapshot_format,
            'url_limit': lambda x: x > 0
        }
    }

    def load_data(self):
        raise NotImplementedError(
            'CommonCrawl data load strategy is not implemented')
