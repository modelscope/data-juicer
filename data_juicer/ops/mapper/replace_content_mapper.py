from typing import List, Union

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('replace_content_mapper')
class ReplaceContentMapper(Mapper):
    """Mapper to replace all content in the text that matches
    a specific regular expression pattern with a designated
    replacement string."""

    def __init__(self,
                 pattern: Union[str, List[str], None] = None,
                 repl: Union[str, List[str]] = '',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param pattern: regular expression pattern(s) to search for within text
        :param repl: replacement string(s), default is empty string
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = pattern
        self.repl = repl
        self.compiled_patterns = []
        if isinstance(pattern, str):
            self.compiled_patterns.append(self._prepare_pattern(pattern))
        elif isinstance(pattern, list):
            for p in pattern:
                self.compiled_patterns.append(self._prepare_pattern(p))

    def _prepare_pattern(self, pattern: str) -> re.Pattern:
        """Prepare the regular expression pattern."""
        if ((pattern is not None and len(pattern) > 2)
                and (pattern.startswith("r'") and pattern.endswith("'")
                     or pattern.startswith('r"') and pattern.endswith('"'))):
            pattern = pattern[2:-1]
        return re.compile(pattern, flags=re.DOTALL)

    def process(self, sample):
        if self.pattern is None:
            return sample

        for i, pattern in enumerate(self.compiled_patterns):
            if isinstance(self.repl, list) and i < len(self.repl):
                replacement = self.repl[i]
            elif isinstance(self.repl, list) and i >= len(self.repl):
                raise ValueError(f"pattern length: {len(self.pattern)} '"
                                 f'must be equal to '
                                 f'repl length: {len(self.repl)}')
            else:
                replacement = self.repl

            sample[self.text_key] = pattern.sub(replacement,
                                                sample[self.text_key])

        return sample
