# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from jsonargparse.typing import ClosedUnitInterval

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter
from ..common import SPECIAL_CHARACTERS


@OPERATORS.register_module('special_characters_filter')
class SpecialCharactersFilter(Filter):
    """Filter to keep samples with special-char ratio within a specific
    range."""

    _batched_op = True

    def __init__(self,
                 min_ratio: ClosedUnitInterval = 0.0,
                 max_ratio: ClosedUnitInterval = 0.25,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_ratio: The min filter ratio in this op, samples will
            be filtered if their special-char ratio is below this
            parameter.
        :param max_ratio: The max filter ratio in this op, samples will
            be filtered if their special-char ratio exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def compute_stats(self, samples):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]

        for i, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.special_char_ratio in stat:
                continue
            # get ratio of special characters
            samples_stats[i][StatsKeys.special_char_ratio] = (
                len([c for c in samples_list[i] if c in SPECIAL_CHARACTERS]) /
                len(samples_list[i])) if len(samples_list[i]) != 0 else 0.0

        return samples

    def process(self, samples):
        if isinstance(samples[Fields.stats], list):
            bool_results = []
            for stat in samples[Fields.stats]:
                if self.min_ratio <= stat[
                        StatsKeys.special_char_ratio] <= self.max_ratio:
                    bool_results.append(True)
                else:
                    bool_results.append(False)
            return bool_results
        else:
            # single sample for ray filter
            if self.min_ratio <= \
                    samples[Fields.stats][StatsKeys.special_char_ratio] \
                    <= self.max_ratio:
                return True
            else:
                return False
