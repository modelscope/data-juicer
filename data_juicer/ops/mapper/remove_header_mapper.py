# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/
# --------------------------------------------------------

import regex as re

from ..base_op import OPERATORS, Mapper


# TODO
@OPERATORS.register_module('remove_header_mapper')
class RemoveHeaderMapper(Mapper):
    """Mapper to remove headers at the beginning of documents in Latex
    samples."""

    def __init__(self, drop_no_head: bool = True, *args, **kwargs):
        """
        Initialization method.

        :param drop_no_head: whether to drop sample texts without
            headers.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = r'^(.*?)('
        self.pattern += r'\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|'
        self.pattern += r'\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|'
        self.pattern += r'\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|'
        self.pattern += r'\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|'
        self.pattern += r'\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|'
        self.pattern += r'\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}'
        self.pattern += r'\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}'
        self.pattern += r')'

        self.drop_no_head = drop_no_head

    def process(self, sample):

        if not re.search(self.pattern, sample[self.text_key], flags=re.DOTALL):
            if self.drop_no_head:
                sample[self.text_key] = ''
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=r'\2',
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
