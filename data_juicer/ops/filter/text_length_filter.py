import sys

from jsonargparse.typing import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('text_length_filter')
class TextLengthFilter(Filter):
    """Filter to keep samples with total text length within a specific
    range."""

    def __init__(self,
                 min_len: PositiveInt = 10,
                 max_len: PositiveInt = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_len: The min text length in the filtering. samples
            will be filtered if their text length is below this
            parameter.
        :param max_len: The max text length in the filtering. samples
            will be filtered if their text length exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_len = min_len
        self.max_len = max_len

    def compute_stats(self, sample):
        # check if it's computed already
        if StatsKeys.text_len in sample[Fields.stats]:
            return sample

        sample[Fields.stats][StatsKeys.text_len] = len(sample[self.text_key])
        return sample

    def process(self, sample):
        if self.min_len <= sample[Fields.stats][
                StatsKeys.text_len] <= self.max_len:
            return True
        else:
            return False
