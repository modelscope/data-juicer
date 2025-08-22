import numbers
from typing import List

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module("tags_specified_field_selector")
class TagsSpecifiedFieldSelector(Selector):
    """Selector to select samples based on the tags of specified
    field."""

    def __init__(self, field_key: str = "", target_tags: List[str] = None, *args, **kwargs):
        """
        Initialization method.

        :param field_key: Selector based on the specified value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param target_tags: Target tags to be select.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_key = field_key
        self.target_tags = set(target_tags)

    def process(self, dataset):
        if len(dataset) <= 1 or not self.field_key:
            return dataset

        field_keys = self.field_key.split(".")
        assert field_keys[0] in dataset.features.keys(), "'{}' not in {}".format(field_keys[0], dataset.features.keys())

        selected_index = []
        for i, item in enumerate(dataset[field_keys[0]]):
            field_value = item
            for key in field_keys[1:]:
                assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
                field_value = field_value[key]
            assert (
                field_value is None or isinstance(field_value, str) or isinstance(field_value, numbers.Number)
            ), "The {} item is not String, Numbers or NoneType".format(i)
            if field_value in self.target_tags:
                selected_index.append(i)

        return dataset.select(selected_index)
