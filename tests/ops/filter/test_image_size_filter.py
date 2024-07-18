import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.image_size_filter import ImageSizeFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageSizeFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    def _run_image_size_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=[op.image_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_min_max(self):

        ds_list = [
            {
                'images': [self.img1_path]  # 171KB
            },
            {
                'images': [self.img2_path]  # 189KB
            },
            {
                'images': [self.img3_path]  # 114KB
            }
        ]
        tgt_list = [{'images': [self.img1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageSizeFilter(min_size='120kb', max_size='180KB')
        self._run_image_size_filter(dataset, tgt_list, op)

    def test_min(self):

        ds_list = [
            {
                'images': [self.img1_path]  # 171KB
            },
            {
                'images': [self.img2_path]  # 189KB
            },
            {
                'images': [self.img3_path]  # 114KB
            }
        ]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageSizeFilter(min_size='120kib')
        self._run_image_size_filter(dataset, tgt_list, op)

    def test_max(self):

        ds_list = [
            {
                'images': [self.img1_path]  # 171KB
            },
            {
                'images': [self.img2_path]  # 189KB
            },
            {
                'images': [self.img3_path]  # 114KB
            }
        ]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageSizeFilter(max_size='180KiB')
        self._run_image_size_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageSizeFilter(min_size='120kb',
                             max_size='180KB',
                             any_or_all='any')
        self._run_image_size_filter(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = ImageSizeFilter(min_size='120kb',
                             max_size='180KB',
                             any_or_all='all')
        self._run_image_size_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
