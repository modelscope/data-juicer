# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from jsonargparse.typing import ClosedUnitInterval, PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import (SPECIAL_CHARACTERS, get_words_from_document,
                      words_refinement)
from ..op_fusion import INTER_WORDS

OP_NAME = 'word_repetition_filter'

with AvailabilityChecking(['sentencepiece'], OP_NAME):
    import sentencepiece  # noqa: F401


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
class WordRepetitionFilter(Filter):
    """Filter to keep samples with word-level n-gram repetition ratio within a
    specific range."""

    def __init__(self,
                 lang: str = 'en',
                 tokenization: bool = False,
                 rep_len: PositiveInt = 10,
                 min_ratio: ClosedUnitInterval = 0.0,
                 max_ratio: ClosedUnitInterval = 0.5,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: sample in which language.
        :param tokenization: whether to use model to tokenize documents
        :param rep_len: Repetition length for word-level n-gram.
        :param min_ratio: The min filter ratio in this op, samples will
            be filtered if their word-level n-gram repetition ratio is
            below this parameter.
        :param max_ratio: The max filter ratio in this op, samples will
            be filtered if their word-level n-gram repetition ratio
            exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.n = rep_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.model_key = None
        self.lang = lang

        if tokenization:
            self.model_key = prepare_model(lang=lang,
                                           model_type='sentencepiece')

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.word_rep_ratio in sample[Fields.stats]:
            return sample

        # try to get words from context
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

        # try to get refined words from context
        refined_words_key = f'{InterVars.refined_words}-True-SPECIAL_CHARS-' \
                            f'False-[2]-'
        if context and refined_words_key in sample[Fields.context]:
            words = sample[Fields.context][refined_words_key]
        else:
            words = words_refinement(words,
                                     lower_case=True,
                                     strip_chars=SPECIAL_CHARACTERS)
            if context:
                sample[Fields.context][refined_words_key] = words
        word_ngrams = [
            ' '.join(words[i:i + self.n])
            for i in range(len(words) - self.n + 1)
        ]
        freq_word_ngrams = {}
        for word_ngram in word_ngrams:
            freq_word_ngrams[word_ngram] = (
                freq_word_ngrams.get(word_ngram, 0) + 1)

        if len(freq_word_ngrams) == 0:
            sample[Fields.stats][StatsKeys.word_rep_ratio] = 0.0
            return sample

        freq_word_ngrams = list(freq_word_ngrams.values())
        rep_more_than_one = [freq for freq in freq_word_ngrams if freq > 1]
        sample[Fields.stats][StatsKeys.word_rep_ratio] = (
            sum(rep_more_than_one) /
            sum(freq_word_ngrams)) if sum(freq_word_ngrams) != 0 else 0.0
        return sample

    def process(self, sample):
        if self.min_ratio <= sample[Fields.stats][StatsKeys.word_rep_ratio] \
                <= self.max_ratio:
            return True
        else:
            return False
