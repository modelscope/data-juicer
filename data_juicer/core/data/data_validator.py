from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Union

from data_juicer.core.data.dj_dataset import NestedDataset
from data_juicer.core.data.ray_dataset import RayDataset


class DataValidator(ABC):
    """Base class for data validation"""

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def validate(self, dataset) -> None:
        """
        Validate dataset content

        Args:
            dataset: The dataset to validate

        Raises:
            DataValidationError: If validation fails
        """
        pass


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class DataValidatorRegistry:
    """Registry for data validators"""

    _validators: Dict[str, Type[DataValidator]] = {}

    @classmethod
    def register(cls, validator_type: str):

        def decorator(validator_class: Type[DataValidator]):
            cls._validators[validator_type] = validator_class
            return validator_class

        return decorator

    @classmethod
    def get_validator(cls,
                      validator_type: str) -> Optional[Type[DataValidator]]:
        return cls._validators.get(validator_type)


@DataValidatorRegistry.register('conversation')
class ConversationDataValidator(DataValidator):
    """Validator for conversation data"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # Validation rules specific to conversation data
        self.required_columns = ['text']
        self.min_turns = config.get('min_turns', 2)
        self.max_turns = config.get('max_turns', 100)

    def validate(self, dataset) -> None:
        # Check required columns
        if not all(col in dataset.column_names
                   for col in self.required_columns):
            raise DataValidationError(
                f'Missing required columns: {self.required_columns}')

        # Validate conversation structure
        for item in dataset:
            turns = self._parse_turns(item['text'])
            if not (self.min_turns <= len(turns) <= self.max_turns):
                raise DataValidationError(
                    f'Conversation must have between {self.min_turns} and '
                    f'{self.max_turns} turns')

            # Additional conversation-specific validations...


@DataValidatorRegistry.register('code')
class CodeDataValidator(DataValidator):
    """Validator for code data"""

    def __init__(self, config: Dict):
        super().__init__(config)

        self.required_columns = ['code', 'language']
        self.supported_languages = config.get('supported_languages', [])

    def validate(self, dataset) -> None:
        # Implement code-specific validation logic...
        pass


@DataValidatorRegistry.register('required_fields')
class RequiredFieldsValidator(DataValidator):
    """Validator that checks for required fields in dataset"""

    def __init__(self, config: Dict):
        """
        Initialize validator with config

        Args:
            config: Dict containing:
                - required_fields: List of field names that must exist
                - field_types: Optional map of field names to expected types
                - allow_missing: Optional float for max ratio missing allowed
        """
        super().__init__(config)

        self.required_fields = config['required_fields']
        self.field_types = config.get('field_types', {})
        # Default no missing allowed
        self.allow_missing = config.get('allow_missing', 0.0)

    def validate(self, dataset: Union[NestedDataset, RayDataset]) -> None:
        """
        Validate dataset has required fields with correct types

        Args:
            dataset: NestedDataset or RayDataset to validate

        Raises:
            DataValidationError: If validation fails
        """
        # Check if fields exist in dataset
        if isinstance(dataset, NestedDataset):
            available_fields = set(dataset.column_names)
        elif isinstance(dataset, RayDataset):
            available_fields = set(dataset.data.schema().names)
        else:
            raise DataValidationError(
                f'Unsupported dataset type: {type(dataset)}')

        missing_fields = set(self.required_fields) - available_fields
        if missing_fields:
            raise DataValidationError(
                f'Dataset missing required fields: {missing_fields}')

        # Check field types and missing values
        for field in self.required_fields:
            # Get expected type if specified
            expected_type = self.field_types.get(field)

            # Sample data for validation
            # For large datasets, we check a sample for performance
            MAX_SAMPLE_SIZE = 1000
            if isinstance(dataset, NestedDataset):
                sample_size = min(MAX_SAMPLE_SIZE, len(dataset))
                sample = dataset.take(sample_size)
                values = sample[field]
            elif isinstance(dataset, RayDataset):  # RayDataset
                sample_size = min(MAX_SAMPLE_SIZE, dataset.data.count())
                sample = dataset.data.take(sample_size)
                values = [row[field] for row in sample]
            else:
                raise NotImplementedError(
                    f'Unsupported dataset type: {type(dataset)}')

            # Check for missing values
            missing_count = sum(1 for v in values if v is None)
            missing_ratio = missing_count / len(values)
            if missing_ratio > self.allow_missing:
                raise DataValidationError(
                    f"Field '{field}' has {missing_ratio:.1%} missing values, "
                    f'exceeding allowed {self.allow_missing:.1%}')

            # Check types if specified
            if expected_type:
                invalid_types = [
                    type(v) for v in values
                    if v is not None and not isinstance(v, expected_type)
                ]
                if invalid_types:
                    raise DataValidationError(
                        f"Field '{field}' contains values of incorrect type. "
                        f'Expected {expected_type.__name__}, '
                        f'got {set(t.__name__ for t in invalid_types)}')
