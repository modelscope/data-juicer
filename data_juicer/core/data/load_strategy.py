import fnmatch
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, Optional, Type, Union

from data_juicer.core.data import DJDataset, RayDataset
from data_juicer.download.downloader import validate_snapshot_format
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


class ConfigValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class ConfigValidator:
    """Mixin class for configuration validation"""

    # Define validation rules for each strategy type
    CONFIG_VALIDATION_RULES = {
        'required_fields': [],  # Fields that must be present
        'optional_fields': [],  # Fields that are optional
        'field_types': {},  # Expected types for fields
        'custom_validators': {}  # Custom validation functions
    }

    def validate_config(self, ds_config: Dict) -> None:
        """
        Validate the configuration dictionary.

        Args:
            ds_config: Configuration dictionary to validate

        Raises:
            ValidationError: If validation fails
        """
        # Check required fields
        missing_fields = [
            field for field in self.CONFIG_VALIDATION_RULES['required_fields']
            if field not in ds_config
        ]
        if missing_fields:
            raise ConfigValidationError(
                f"Missing required fields: {', '.join(missing_fields)}")

        # Optional fields
        # no need for any special checks

        # Check field types
        for field, expected_type in self.CONFIG_VALIDATION_RULES[
                'field_types'].items():
            if field in ds_config:
                value = ds_config[field]
                if not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Field '{field}' must be of "
                        "type '{expected_type.__name__}', "
                        f"got '{type(value).__name__}'")

        # Run custom validators
        for field, validator in self.CONFIG_VALIDATION_RULES[
                'custom_validators'].items():
            if field in ds_config:
                try:
                    validator(ds_config[field])
                except Exception as e:
                    raise ConfigValidationError(
                        f"Validation failed for field '{field}': {str(e)}")


class DataValidator:
    """Mixin class for data content validation"""

    # Define data validation rules
    DATA_VALIDATION_RULES = {
        'required_columns': [],  # Columns that must be present in the dataset
        'column_types': {},  # Expected types for columns
        'custom_validators': {}  # Custom validation functions for data content
    }

    def validate_data(self, dataset) -> None:
        """
        Validate the actual dataset content.

        Args:
            dataset: The loaded dataset to validate

        Raises:
            DataValidationError: If validation fails
        """
        # Check required columns
        if hasattr(dataset, 'column_names'):
            missing_columns = [
                col for col in self.DATA_VALIDATION_RULES['required_columns']
                if col not in dataset.column_names
            ]
            if missing_columns:
                raise DataValidationError(
                    f"Missing required columns: {', '.join(missing_columns)}")

        # Check column types
        for col, expected_type in self.DATA_VALIDATION_RULES[
                'column_types'].items():
            if col in dataset.column_names:
                # Sample check for performance
                sample = dataset.select(range(min(100, len(dataset))))
                if not all(
                        isinstance(val, expected_type) for val in sample[col]):
                    raise DataValidationError(
                        f"Column '{col}' contains values of incorrect type")

        # Run custom validators
        for validator_name, validator in self.DATA_VALIDATION_RULES[
                'custom_validators'].items():
            try:
                validator(dataset)
            except Exception as e:
                raise DataValidationError(
                    f"Data validation '{validator_name}' failed: {str(e)}")


class DataLoadStrategy(ABC, ConfigValidator):
    """
    abstract class for data load strategy
    """

    def __init__(self, ds_config: Dict):
        self.validate_config(ds_config)
        self.ds_config = ds_config

    @abstractmethod
    def load_data(self, cfg: Namespace) -> Union[DJDataset, RayDataset]:
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
    def load_data(self) -> RayDataset:
        pass


class LocalDataLoadStrategy(DataLoadStrategy):
    """
    abstract class for data load strategy for LocalExecutor
    """

    @abstractmethod
    def load_data(self, cfg: Namespace) -> DJDataset:
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


@DataLoadStrategyRegistry.register('ray', 'ondisk', 'json')
class RayOndiskJsonDataLoadStrategy(RayDataLoadStrategy):

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': (str, list)  # Can be string or list
        },
        'custom_validators': {}
    }

    def load_data(self, cfg: Namespace):
        return rd.read_json(self.ds_config.path)


@DataLoadStrategyRegistry.register('ray', 'remote', 'huggingface')
class RayHuggingfaceDataLoadStrategy(RayDataLoadStrategy):

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': (str, list)  # Can be string or list
        },
        'custom_validators': {}
    }

    def load_data(self, cfg: Namespace):
        pass


@DataLoadStrategyRegistry.register('local', 'ondisk', '*')
class LocalOndiskDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for on disk data for LocalExecutor
    rely on AutoFormatter for actual data loading
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': (str, list)  # Can be string or list
        },
        'custom_validators': {}
    }

    def load_data(self, cfg: Namespace):
        pass


@DataLoadStrategyRegistry.register('local', 'remote', 'huggingface')
class LocalHuggingfaceDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for Huggingface dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': (str, list)  # Can be string or list
        },
        'custom_validators': {}
    }

    def load_data(self, cfg: Namespace):
        pass


@DataLoadStrategyRegistry.register('local', 'remote', 'modelscope')
class LocalModelScopeDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for ModelScope dataset for LocalExecutor
    """

    def load_data(self, cfg: Namespace):
        pass


@DataLoadStrategyRegistry.register('local', 'remote', 'arxiv')
class LocalArxivDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for arxiv dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': (str, list)  # Can be string or list
        },
        'custom_validators': {}
    }

    def load_data(self, cfg: Namespace):
        pass


@DataLoadStrategyRegistry.register('local', 'remote', 'wiki')
class LocalWikiDataLoadStrategy(LocalDataLoadStrategy):
    """
    data load strategy for wiki dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        'required_fields': ['path'],
        'field_types': {
            'path': (str, list)  # Can be string or list
        },
        'custom_validators': {}
    }

    def load_data(self, cfg: Namespace):
        pass


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

    def load_data(self, cfg: Namespace):
        pass
