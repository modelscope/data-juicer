import os
import unittest

from data_juicer.format.mixture_formatter import MixtureFormatter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class MixtureFormatterTest(DataJuicerTestCaseBase):

    def setUp(self):
        self._path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'data', 'structured')
        self._file = os.path.join(self._path, 'demo-dataset.jsonl')
        self._file2 = self._file

    def test_only_file(self):
        formatter = MixtureFormatter(self._file)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_sample_weight(self):
        formatter = MixtureFormatter('0.5 ' + self._file)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 3)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_sample_number(self):
        max_samples = 2
        formatter = MixtureFormatter(self._file, max_samples=max_samples)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), max_samples)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_sample_number_weight(self):
        max_samples = 2
        formatter = MixtureFormatter('0.5 ' + self._file,
                                     max_samples=max_samples)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), max_samples)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_multi_datasets_without_weight(self):
        data_path = self._file + ' ' + self._file2
        formatter = MixtureFormatter(data_path)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 12)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_multi_datasets_with_one_weight(self):
        data_path = '0.5 ' + self._file + ' ' + self._file2
        formatter = MixtureFormatter(data_path)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 9)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_multi_datasets_with_weight(self):
        data_path = '0.5 ' + self._file + ' 0.5 ' + self._file2
        formatter = MixtureFormatter(data_path)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_multi_datasets_with_sample(self):
        max_samples = 7
        data_path = '0.5 ' + self._file + ' 0.5 ' + self._file2
        formatter = MixtureFormatter(data_path, max_samples=max_samples)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), max_samples)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])


if __name__ == '__main__':
    unittest.main()
