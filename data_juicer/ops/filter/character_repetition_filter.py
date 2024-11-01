# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('character_repetition_filter')
class CharacterRepetitionFilter(Filter):
    """Filter to keep samples with char-level n-gram repetition ratio within a
    specific range."""

    _batched_op = True

    def __init__(self,
                 rep_len: PositiveInt = 10,
                 min_ratio: float = 0.0,
                 max_ratio: float = 0.5,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param rep_len: Repetition length for char-level n-gram.
        :param min_ratio: The min filter ratio in this op, samples will
            be filtered if their char-level n-gram repetition ratio is
            below this parameter.
        :param max_ratio: The max filter ratio in this op, samples will
            be filtered if their char-level n-gram repetition ratio
            exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.n = rep_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def compute_stats_batched(self, samples):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]

        for idx, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.char_rep_ratio in stat:
                continue

            cur_text = samples_list[idx]
            char_ngrams = [
                cur_text[i:i + self.n]
                for i in range(len(cur_text) - self.n + 1)
            ]
            freq_char_ngrams = {}
            for char_ngram in char_ngrams:
                freq_char_ngrams[char_ngram] = (
                    freq_char_ngrams.get(char_ngram, 0) + 1)

            if len(freq_char_ngrams) == 0:
                samples_stats[idx][StatsKeys.char_rep_ratio] = 0.0
                continue

            freq_char_ngrams = sorted(list(freq_char_ngrams.values()),
                                      reverse=True)
            num_no_rep_char_ngrams = len(
                [el for el in freq_char_ngrams if el == 1])
            num_rep_char_ngrams = min(
                int(np.sqrt(len(freq_char_ngrams))),
                len(freq_char_ngrams) - num_no_rep_char_ngrams,
            )
            samples_stats[idx][StatsKeys.char_rep_ratio] = (
                sum(freq_char_ngrams[:num_rep_char_ngrams]) /
                sum(freq_char_ngrams)) if sum(freq_char_ngrams) != 0 else 0.0

        return samples

    def process_batched(self, samples):
        if isinstance(samples[Fields.stats], list):
            return map(
                lambda stat: self.min_ratio <= stat[StatsKeys.char_rep_ratio]
                <= self.max_ratio, samples[Fields.stats])
        else:
            # single sample for ray filter
            if self.min_ratio <= samples[Fields.stats][
                    StatsKeys.char_rep_ratio] <= self.max_ratio:
                return True
            else:
                return False
