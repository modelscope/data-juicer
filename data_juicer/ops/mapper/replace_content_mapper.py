import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('replace_content_mapper')
class ReplaceContentMapper(Mapper):
    """Mapper to replace all content in the text that matches
    a specific regular expression pattern with a designated
    replacement string."""

    def __init__(self, pattern: str = None, repl: str = '', *args, **kwargs):
        """
        Initialization method.

        :param pattern: regular expression pattern to search for within text.
        :param repl: replacement string, default is empty string.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = pattern
        if ((pattern is not None and len(pattern) > 2)
                and (pattern.startswith("r'") and pattern.endswith("'")
                     or pattern.startswith('r"') and pattern.endswith('"'))):
            self.pattern = pattern[2:-1]
        self.repl = repl

    def process(self, sample):

        if self.pattern is None:
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=self.repl,
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
