# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

import sys

from ..base_op import OPERATORS, Mapper
from ..common import (SPECIAL_CHARACTERS, merge_on_whitespace_tab_newline,
                      split_on_newline_tab_whitespace, strip)


@OPERATORS.register_module('remove_long_words_mapper')
class RemoveLongWordsMapper(Mapper):
    """Mapper to remove long words within a specific range."""

    def __init__(self,
                 min_len: int = 1,
                 max_len: int = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_len: The min mapper word length in this op, words
            will be filtered if their length is below this parameter.
        :param max_len: The max mapper word length in this op, words
            will be filtered if their length exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_len = min_len
        self.max_len = max_len

    def should_keep_long_word(self, word):
        if self.min_len <= len(word) <= self.max_len:
            return True
        elif self.min_len <= len(strip(word,
                                       SPECIAL_CHARACTERS)) <= self.max_len:
            return True
        else:
            return False

    def process(self, sample):

        sentences = split_on_newline_tab_whitespace(sample[self.text_key])
        sentences = [[[
            word for word in subsentence if self.should_keep_long_word(word)
        ] for subsentence in sentence] for sentence in sentences]
        sample[self.text_key] = merge_on_whitespace_tab_newline(sentences)
        return sample
