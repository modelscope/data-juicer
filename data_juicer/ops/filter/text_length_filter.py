import sys

from jsonargparse.typing import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('text_length_filter')
class TextLengthFilter(Filter):
    """Filter to keep samples with total text length within a specific
    range."""

    _batched_op = True

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

    def compute_stats(self, samples):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        for i, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.text_len in stat:
                continue
            else:
                samples_stats[i][StatsKeys.text_len] = len(samples_list[i])

        return samples

    def process(self, samples):
        if isinstance(samples[Fields.stats], list):
            bool_res = []
            for stat in samples[Fields.stats]:
                if self.min_len <= stat[StatsKeys.text_len] <= self.max_len:
                    bool_res.append(True)
                else:
                    bool_res.append(False)
            return bool_res
        else:
            # single sample for ray filter
            if self.min_len <= samples[Fields.stats][
                    StatsKeys.text_len] <= self.max_len:
                return True
            else:
                return False
