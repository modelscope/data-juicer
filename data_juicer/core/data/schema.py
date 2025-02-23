from dataclasses import dataclass
from typing import Any, Dict, List

import pyarrow as pa
from datasets import Array2D, Array3D, ClassLabel, Sequence, Value


@dataclass
class Schema:
    """Dataset schema representation.

    Attributes:
        column_types: Mapping of column names to their types
        columns: List of column names in order
    """
    column_types: Dict[str, Any]
    columns: List[str]

    @classmethod
    def map_hf_type_to_python(cls, feature):
        """Map HuggingFace feature type to Python type.

        Recursively maps nested types (e.g., List[str], Dict[str, int]).

        Examples:
            Value('string') -> str
            Sequence(Value('int32')) -> List[int]
            Dict({'text': Value('string')}) -> Dict[str, Any]

        Args:
            feature: HuggingFace feature type

        Returns:
            Corresponding Python type
        """
        if isinstance(feature, Value):
            # Map Value types
            type_mapping = {
                'string': str,
                'int32': int,
                'int64': int,
                'float32': float,
                'float64': float,
                'bool': bool,
                'binary': bytes
            }
            return type_mapping.get(feature.dtype, Any)

        elif isinstance(feature, (Sequence, Array2D, Array3D)):
            # Handle sequences/lists
            return list

        # Dictionary types - check if it's a dictionary feature
        elif isinstance(feature, dict) or str(type(feature)).endswith('Dict'):
            return dict

        elif isinstance(feature, ClassLabel):
            # Handle class labels
            return int

        else:
            # Default to Any for unknown types
            return Any

    @classmethod
    def map_ray_type_to_python(cls, ray_type: pa.DataType) -> type:
        """Map Ray/Arrow data type to Python type.

        Args:
            ray_type: PyArrow DataType

        Returns:
            Corresponding Python type
        """

        # String types
        if pa.types.is_string(ray_type):
            return str
        if pa.types.is_binary(ray_type):
            return bytes

        # Numeric types
        if pa.types.is_integer(ray_type):
            return int
        if pa.types.is_floating(ray_type):
            return float

        # Boolean
        if pa.types.is_boolean(ray_type):
            return bool

        # List/Array types
        if pa.types.is_list(ray_type):
            return list

        # Dictionary/Struct types
        if pa.types.is_struct(ray_type) or pa.types.is_map(ray_type):
            return dict

        # Fallback
        return Any

    def __post_init__(self):
        """Validate schema after initialization"""
        # Ensure all columns are in column_types
        if not all(col in self.column_types for col in self.columns):
            missing = set(self.columns) - set(self.column_types.keys())
            raise ValueError(
                f'Missing type definitions for columns: {missing}')

    def __str__(self) -> str:
        """Return formatted string representation of schema"""
        lines = ['Dataset Schema:']
        lines.append('-' * 40)
        for col in self.columns:
            lines.append(f'{col}: {self.column_types[col]}')
        return '\n'.join(lines)
