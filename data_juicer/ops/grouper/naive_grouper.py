from ..base_op import OPERATORS, Grouper, convert_list_dict_to_dict_list


@OPERATORS.register_module("naive_grouper")
class NaiveGrouper(Grouper):
    """Group all samples to one batched sample."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def process(self, dataset):
        if len(dataset) == 0:
            return dataset

        batched_sample = convert_list_dict_to_dict_list(dataset)

        return [batched_sample]
