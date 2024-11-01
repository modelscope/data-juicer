import sys

from ..base_op import OPERATORS, Filter


def is_number(s):
    if s:
        try:
            float(s)
            return True
        except ValueError:
            pass
    return False


@OPERATORS.register_module('specified_numeric_field_filter')
class SpecifiedNumericFieldFilter(Filter):
    """
    Filter based on specified numeric field information.

    If the specified numeric information in the sample is not within the
    specified range, the sample will be filtered.
    """

    def __init__(self,
                 field_key: str = '',
                 min_value: float = -sys.maxsize,
                 max_value: float = sys.maxsize,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param field_key: Filter based on the specified numeric value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param min_value: The min filter value in SpecifiedNumericField
            op, samples will be filtered if their specified numeric
            field value is below this parameter.
        :param max_value: The max filter value in SpecifiedNumericField
            op, samples will be filtered if their specified numeric
            field value exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_key = field_key
        self.min_value = min_value
        self.max_value = max_value

    def compute_stats_single(self, sample):
        return sample

    def process_single(self, sample):
        if not self.field_key:
            return True

        field_value = sample
        for key in self.field_key.split('.'):
            assert key in field_value.keys(), "'{}' not in {}".format(
                key, field_value.keys())
            field_value = field_value[key]

        if is_number(field_value):
            field_value = float(field_value)
            return self.min_value <= field_value <= self.max_value
        else:
            return False
