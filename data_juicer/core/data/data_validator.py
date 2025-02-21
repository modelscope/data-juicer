from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from data_juicer.core.data.dj_dataset import DJDataset


class DataValidator(ABC):
    """Base class for data validation"""

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def validate(self, dataset: DJDataset) -> None:
        """
        Validate dataset content

        Args:
            dataset: The dataset to validate

        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(dataset, DJDataset):
            raise DataValidationError(
                'unsupported dataset type; must be a DJDataset')


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


class BaseConversationValidator(DataValidator):
    """Base class for conversation validators"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_turns = config.get('min_turns', 1)
        self.max_turns = config.get('max_turns', 100)
        self.sample_size = config.get('sample_size', 100)

    def validate(self, dataset: DJDataset) -> None:
        """Base validation for all conversation formats"""
        super().validate(dataset)

        schema = dataset.schema()
        if not all(col in schema.columns for col in ['text']):
            raise DataValidationError('Missing required column: text')

        for item in dataset.get_column('text', self.sample_size):
            self.validate_conversation(item)

    @abstractmethod
    def validate_conversation(self, data: Dict) -> None:
        """Validate specific conversation format"""
        pass


@DataValidatorRegistry.register('swift_messages')
class SwiftMessagesValidator(BaseConversationValidator):
    """Validator for Swift Messages format"""

    def validate_conversation(self, data: Dict) -> None:
        if 'messages' not in data:
            raise DataValidationError("Missing 'messages' field")

        messages = data['messages']
        if not isinstance(messages, list):
            raise DataValidationError("'messages' must be an array")

        if not (self.min_turns <= len(messages) <= self.max_turns):
            raise DataValidationError(
                f'Conversation must have between {self.min_turns} and '
                f'{self.max_turns} messages')

        for msg in messages:
            if not isinstance(msg, dict):
                raise DataValidationError('Message must be an object')

            if 'role' not in msg:
                raise DataValidationError("Missing 'role' field in message")
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise DataValidationError(f"Invalid role: {msg['role']}")

            if 'content' not in msg:
                raise DataValidationError("Missing 'content' field in message")
            if not isinstance(msg['content'], str):
                raise DataValidationError("'content' must be string")


@DataValidatorRegistry.register('swift_sharegpt')
class SwiftShareGPTValidator(BaseConversationValidator):
    """Validator for Swift ShareGPT format"""

    def validate_conversation(self, data: Dict) -> None:
        if 'system' in data and not isinstance(data['system'], str):
            raise DataValidationError("'system' must be string")

        if 'conversation' not in data:
            raise DataValidationError("Missing 'conversation' field")

        conv = data['conversation']
        if not isinstance(conv, list):
            raise DataValidationError("'conversation' must be an array")

        if not (self.min_turns <= len(conv) <= self.max_turns):
            raise DataValidationError(
                f'Conversation must have between {self.min_turns} and '
                f'{self.max_turns} turns')

        for turn in conv:
            if not isinstance(turn, dict):
                raise DataValidationError('Turn must be an object')

            if 'human' not in turn or not isinstance(turn['human'], str):
                raise DataValidationError("Missing or invalid 'human' field")
            if 'assistant' not in turn or not isinstance(
                    turn['assistant'], str):
                raise DataValidationError(
                    "Missing or invalid 'assistant' field")


@DataValidatorRegistry.register('alpaca')
class AlpacaValidator(BaseConversationValidator):
    """Validator for Alpaca format"""

    def validate_conversation(self, data: Dict) -> None:
        if 'system' in data and not isinstance(data['system'], str):
            raise DataValidationError("'system' must be string")

        for field in ['instruction', 'input', 'output']:
            if field not in data:
                raise DataValidationError(f"Missing '{field}' field")
            if not isinstance(data[field], str):
                raise DataValidationError(f"'{field}' must be string")


@DataValidatorRegistry.register('swift_query_response')
class SwiftQueryResponseValidator(BaseConversationValidator):
    """Validator for Swift Query-Response format"""

    def validate_conversation(self, data: Dict) -> None:
        if 'system' in data and not isinstance(data['system'], str):
            raise DataValidationError("'system' must be string")

        for field in ['query', 'response']:
            if field not in data:
                raise DataValidationError(f"Missing '{field}' field")
            if not isinstance(data[field], str):
                raise DataValidationError(f"'{field}' must be string")

        if 'history' in data:
            if not isinstance(data['history'], list):
                raise DataValidationError("'history' must be an array")

            total_turns = len(data['history']) + 1
            if not (self.min_turns <= total_turns <= self.max_turns):
                raise DataValidationError(
                    f'Conversation must have between {self.min_turns} and '
                    f'{self.max_turns} turns including history')

            for turn in data['history']:
                if not isinstance(turn, list) or len(turn) != 2:
                    raise DataValidationError(
                        'History turn must be [query, response] pair')
                if not all(isinstance(x, str) for x in turn):
                    raise DataValidationError(
                        'History elements must be strings')


@DataValidatorRegistry.register('dj_conversation')
class DataJuicerFormatValidator(BaseConversationValidator):
    """Validator for Data-Juicer default format"""

    def validate_conversation(self, data: Dict) -> None:
        if 'system' in data and not isinstance(data['system'], str):
            raise DataValidationError("'system' must be string")

        for field in ['instruction', 'query', 'response']:
            if field not in data:
                raise DataValidationError(f"Missing '{field}' field")
            if not isinstance(data[field], str):
                raise DataValidationError(f"'{field}' must be string")

        if 'history' in data:
            if not isinstance(data['history'], list):
                raise DataValidationError("'history' must be an array")

            total_turns = len(data['history']) + 1
            if not (self.min_turns <= total_turns <= self.max_turns):
                raise DataValidationError(
                    f'Conversation must have between {self.min_turns} and '
                    f'{self.max_turns} turns including history')

            for turn in data['history']:
                if not isinstance(turn, list) or len(turn) != 2:
                    raise DataValidationError(
                        'History turn must be [query, response] pair')
                if not all(isinstance(x, str) for x in turn):
                    raise DataValidationError(
                        'History elements must be strings')


@DataValidatorRegistry.register('code')
class CodeDataValidator(DataValidator):
    """Validator for code data"""

    def __init__(self, config: Dict):
        super().__init__(config)

        self.required_columns = ['code', 'language']
        self.supported_languages = config.get('supported_languages', [])

    def validate(self, dataset: DJDataset) -> None:
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

    def validate(self, dataset: DJDataset) -> None:
        """
        Validate dataset has required fields with correct types

        Args:
            dataset: NestedDataset or RayDataset to validate

        Raises:
            DataValidationError: If validation fails
        """
        super().validate(dataset)

        # Check if fields exist in dataset
        available_fields = set(dataset.schema().columns)

        missing_fields = set(self.required_fields) - available_fields
        if missing_fields:
            raise DataValidationError(
                f'Dataset missing required fields: {missing_fields}')

        # Check field types and missing values
        for field in self.required_fields:
            # Get expected type if specified
            expected_type = self.field_types.get(field)

            # Sample head part of data for validation
            MAX_SAMPLE_SIZE = 1000
            sample_values = dataset.get_column(field, MAX_SAMPLE_SIZE)

            # Check for missing values
            missing_count = sum(1 for v in sample_values if v is None)
            missing_ratio = missing_count / len(sample_values)
            if missing_ratio > self.allow_missing:
                raise DataValidationError(
                    f"Field '{field}' has {missing_ratio:.1%} missing values, "
                    f'exceeding allowed {self.allow_missing:.1%}')

            # Check types if specified
            if expected_type:
                invalid_types = [
                    type(v) for v in sample_values
                    if v is not None and not isinstance(v, expected_type)
                ]
                if invalid_types:
                    raise DataValidationError(
                        f"Field '{field}' contains values of incorrect type. "
                        f'Expected {expected_type.__name__}, '
                        f'got {set(t.__name__ for t in invalid_types)}')
