import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('clean_email_mapper')
class CleanEmailMapper(Mapper):
    """Mapper to clean email in text samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = r'[A-Za-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+'

    def process(self, sample):

        if not re.search(self.pattern, sample[self.text_key], flags=re.DOTALL):
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=r'',
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
