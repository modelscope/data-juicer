from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Schema:
    """Dataset schema representation.

    Attributes:
        column_types: Mapping of column names to their types
        columns: List of column names in order
    """
    column_types: Dict[str, Any]
    columns: List[str]

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
