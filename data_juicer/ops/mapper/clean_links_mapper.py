# Some code here has been modified from:
# https://github.com/kallewesterling/CleanText/
# --------------------------------------------------------
import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('clean_links_mapper')
class CleanLinksMapper(Mapper):
    """Mapper to clean links like http/https/ftp in text samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = r'(?i)\b('
        self.pattern += r'(?:[a-z][\w-]+:(?:\/{1,3}|'
        self.pattern += r'[a-z0-9%])|www\d{0,3}[.]|'
        self.pattern += r'[a-z0-9.\-]+[.][a-z]{2,4}\/)'
        self.pattern += r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))'
        self.pattern += r'+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'
        self.pattern += r'[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])'
        self.pattern += r')'

    def process(self, sample):

        if not re.search(self.pattern, sample[self.text_key], flags=re.DOTALL):
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=r'',
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
