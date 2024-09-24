import os
import unittest

from data_juicer.format.empty_formatter import EmptyFormatter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class EmptyFormatterTest(DataJuicerTestCaseBase):

    text_key = 'text'

    def test_empty_dataset(self):
        ds_len = 10
        formatter = EmptyFormatter(length=ds_len, feature_keys=[self.text_key])
        ds = formatter.load_dataset()

        self.assertEqual(len(ds), ds_len)
        self.assertEqual(list(ds.features.keys()), [self.text_key])

        for item in ds:
            self.assertDictEqual(item, {self.text_key: None})

        # test map
        update_column = {self.text_key: 1}

        def map_fn(sample):
            sample.update(update_column)
            return sample

        ds = ds.map(map_fn)
        self.assertEqual(len(ds), ds_len)
        for item in ds:
            self.assertDictEqual(item, update_column)

        # test filter
        def filter_fn(sample):
            return sample[self.text_key] > 2
        
        ds = ds.filter(filter_fn)
        self.assertEqual(len(ds), 0)


if __name__ == '__main__':
    unittest.main()
