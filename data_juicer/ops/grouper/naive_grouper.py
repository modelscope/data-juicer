from ..base_op import OPERATORS, Grouper


@OPERATORS.register_module('naive_grouper')
class NaiveGrouper(Grouper):
    """Group all samples to one batched sample. """

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

        keys = dataset[0].keys()
        batched_sample = {k: [None] * len(dataset) for k in keys}
        for i, sample in enumerate(dataset):
            for k in keys:
                batched_sample[k][i] = sample[k]

        return [batched_sample]
