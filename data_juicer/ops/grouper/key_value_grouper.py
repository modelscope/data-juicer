from typing import List, Optional

from data_juicer.utils.common_utils import dict_to_hash, nested_access

from ..base_op import OPERATORS, Grouper, convert_list_dict_to_dict_list
from .naive_grouper import NaiveGrouper


@OPERATORS.register_module('key_value_grouper')
class KeyValueGrouper(Grouper):
    """Group samples to batched samples according values in given keys. """

    def __init__(self,
                 group_by_keys: Optional[List[str]] = None,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param group_by_keys: group samples according values in the keys.
            Support for nested keys such as "__dj__stats__.text_len".
            It is [self.text_key] in default.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.group_by_keys = group_by_keys or [self.text_key]
        self.naive_grouper = NaiveGrouper()

    def process(self, dataset):

        if len(dataset) == 0:
            return dataset

        sample_map = {}
        for sample in dataset:
            cur_dict = {}
            for key in self.group_by_keys:
                cur_dict[key] = nested_access(sample, key)
            sample_key = dict_to_hash(cur_dict)
            if sample_key in sample_map:
                sample_map[sample_key].append(sample)
            else:
                sample_map[sample_key] = [sample]

        batched_samples = [
            convert_list_dict_to_dict_list(sample_map[k]) for k in sample_map
        ]

        return batched_samples
