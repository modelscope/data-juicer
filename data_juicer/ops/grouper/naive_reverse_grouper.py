from ..base_op import OPERATORS, Grouper, convert_dict_list_to_list_dict


@OPERATORS.register_module('naive_reverse_grouper')
class NaiveReverseGrouper(Grouper):
    """Split batched samples to samples. """

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

        samples = []
        for sample in dataset:
            samples.extend(convert_dict_list_to_list_dict(sample))

        return samples
