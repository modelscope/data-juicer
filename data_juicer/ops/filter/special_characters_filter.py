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

    def compute_stats(self, sample):
        # check if it's computed already
        if StatsKeys.special_char_ratio in sample[Fields.stats]:
            return sample

        # get ratio of special characters
        sample[Fields.stats][StatsKeys.special_char_ratio] = (
            len([c
                 for c in sample[self.text_key] if c in SPECIAL_CHARACTERS]) /
            len(sample[self.text_key])) if len(
                sample[self.text_key]) != 0 else 0.0
        return sample

    def process(self, sample):
        if self.min_ratio <= \
                sample[Fields.stats][StatsKeys.special_char_ratio] \
                <= self.max_ratio:
            return True
        else:
            return False
