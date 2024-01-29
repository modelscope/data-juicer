import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('clean_email_mapper')
class CleanEmailMapper(Mapper):
    """Mapper to clean email in text samples."""

    def __init__(self, pattern: str = None, repl: str = '', *args, **kwargs):
        """
        Initialization method.

        :param pattern: regular expression pattern to search for within text.
        :param repl: replacement string, default is empty string.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if pattern is None:
            self.pattern = r'[A-Za-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+'
        else:
            self.pattern = pattern
            if ((len(pattern) > 2) and
                (pattern.startswith("r'") and pattern.endswith("'")
                 or pattern.startswith('r"') and pattern.endswith('"'))):
                self.pattern = pattern[2:-1]

        self.repl = repl

    def process(self, sample):

        if not re.search(self.pattern, sample[self.text_key], flags=re.DOTALL):
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=self.repl,
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
