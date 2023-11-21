import sys

from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import (SPECIAL_CHARACTERS, get_words_from_document,
                      words_refinement)
from ..op_fusion import INTER_WORDS

OP_NAME = 'words_num_filter'

with AvailabilityChecking(['sentencepiece'], OP_NAME):
    import sentencepiece  # noqa: F401


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
class WordNumFilter(Filter):
    """Filter to keep samples with total words number within a specific
    range."""

    def __init__(self,
                 lang: str = 'en',
                 tokenization: bool = False,
                 min_num: PositiveInt = 10,
                 max_num: PositiveInt = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: sample in which language.
        :param tokenization: whether to use model to tokenize documents
        :param min_num: The min filter word number in this op, samples
            will be filtered if their word number is below this
            parameter.
        :param max_num: The max filter word number in this op, samples
            will be filtered if their word number exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_num = min_num
        self.max_num = max_num
        self.model_key = None
        self.lang = lang

        if tokenization:
            self.model_key = prepare_model(lang=lang,
                                           model_type='sentencepiece')

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.num_words in sample[Fields.stats]:
            return sample

        words_key = f'{InterVars.words}-{self.model_key}'
        if context and words_key in sample[Fields.context]:
            words = sample[Fields.context][words_key]
        else:
            tokenizer = get_model(self.model_key,
                                  lang=self.lang,
                                  model_type='sentencepiece')
            words = get_words_from_document(
                sample[self.text_key],
                token_func=tokenizer.encode_as_pieces if tokenizer else None)
            if context:
                sample[Fields.context][words_key] = words
        words = words_refinement(words, strip_chars=SPECIAL_CHARACTERS)
        sample[Fields.stats][StatsKeys.num_words] = len(words)
        return sample

    def process(self, sample):
        if self.min_num <= sample[Fields.stats][
                StatsKeys.num_words] <= self.max_num:
            return True
        else:
            return False
