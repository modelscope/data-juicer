from typing import List

from data_juicer.utils.constant import Fields

from ..base_op import NON_STATS_FILTERS, OPERATORS, Filter

OP_NAME = "specified_field_filter"


@NON_STATS_FILTERS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class SpecifiedFieldFilter(Filter):
    """
    Filter based on specified field information.

    If the specified field information in the sample is not within the
    specified target value, the sample will be filtered.
    """

    def __init__(self, field_key: str = "", target_value: List = [], *args, **kwargs):
        """
        Initialization method.

        :param field_key: Filter based on the specified value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param target_value: The range of specified field information
            corresponding to the samples that need to be retained.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_key = field_key
        self.target_value = target_value

    def compute_stats_single(self, sample):
        # get the value from the original field
        field_value = sample
        for key in self.field_key.split("."):
            assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
            field_value = field_value[key]
        # copy it into the stats field
        if self.field_key not in sample[Fields.stats]:
            sample[Fields.stats][self.field_key] = field_value
        return sample

    def process_single(self, sample):
        if not (self.field_key and self.target_value):
            return True

        field_value = sample[Fields.stats][self.field_key]

        if not (isinstance(field_value, list) or isinstance(field_value, tuple)):
            field_value = [field_value]
        res_bool = True
        for value in field_value:
            if value not in self.target_value:
                res_bool = False
        if self.reversed_range:
            res_bool = not res_bool
        return res_bool
