import unittest

from datasets import Dataset

from data_juicer.ops.filter.random_sample_filter import RandomSampleFilter
from data_juicer.utils.constant import Fields


class RandomSampleFilterTest(unittest.TestCase):

    def _run_random_sample_filter(self, dataset: Dataset, op):
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertIsNotNone(res_list)
        print("Processed data after the operator:", res_list)

    def test_random_sample_case(self):
        ds_list = [
            {'text': 'Sample text 1'},
            {'text': 'Sample text 2'},
            {'text': 'Sample text 3'},
        ]
        dataset = Dataset.from_list(ds_list)
        op = RandomSampleFilter(sample_percentage=0.9)
        self._run_random_sample_filter(dataset, op)


if __name__ == '__main__':
    unittest.main()

