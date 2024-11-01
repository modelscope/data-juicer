# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/
# --------------------------------------------------------

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('remove_header_mapper')
class RemoveHeaderMapper(Mapper):
    """Mapper to remove headers at the beginning of documents in Latex
    samples."""

    _batched_op = True

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

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            if not re.search(self.pattern, text, flags=re.DOTALL):
                if self.drop_no_head:
                    text = ''
                continue
            text = re.sub(pattern=self.pattern,
                          repl=r'\2',
                          string=text,
                          flags=re.DOTALL)

            samples[self.text_key][idx] = text

        return samples
