import heapq
from typing import Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.common_utils import stats_to_number

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module("topk_specified_field_selector")
class TopkSpecifiedFieldSelector(Selector):
    """Selector to select top samples based on the sorted specified field
    value."""

    def __init__(
        self,
        field_key: str = "",
        top_ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
        topk: Optional[PositiveInt] = None,
        reverse: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param field_key: Selector based on the specified value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param top_ratio: Ratio of selected top samples, samples will be
            selected if their specified field values are within this
            parameter. When both topk and top_ratio are set, the value
            corresponding to the smaller number of samples will be
            applied.
        :param topk: Number of selected top sample, samples will be
            selected if their specified field values are within this
            parameter. When both topk and top_ratio are set, the value
            corresponding to the smaller number of samples will be
            applied.
        :param reverse: Determine the sorting rule, if reverse=True,
            then sort in descending order.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_key = field_key
        self.top_ratio = top_ratio
        self.topk = topk
        self.reverse = reverse

    def process(self, dataset):
        if len(dataset) <= 1 or not self.field_key:
            return dataset

        select_num = 0
        if not self.top_ratio:
            if not self.topk:
                return dataset
            else:
                select_num = self.topk
        else:
            select_num = self.top_ratio * len(dataset)
            if self.topk and self.topk < select_num:
                select_num = self.topk

        field_keys = self.field_key.split(".")
        assert field_keys[0] in dataset.features.keys(), "'{}' not in {}".format(field_keys[0], dataset.features.keys())

        if len(field_keys) == 1:
            field_value_list = dataset[field_keys[0]]
        else:
            field_value_list = []
            for item in dataset[field_keys[0]]:
                field_value = item
                for key in field_keys[1:]:
                    assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
                    field_value = field_value[key]
                field_value_list.append(stats_to_number(field_value, self.reverse))

        if self.reverse:
            select_index = heapq.nlargest(int(select_num), range(len(dataset)), field_value_list.__getitem__)
        else:
            select_index = heapq.nsmallest(int(select_num), range(len(dataset)), field_value_list.__getitem__)
        return dataset.select(select_index)
