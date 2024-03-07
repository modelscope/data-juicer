import os
import unittest

from datasets import Dataset

from data_juicer.ops.filter.face_area_filter import FaceAreaFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class FaceAreaFilterTest(DataJuicerTestCaseBase):

    maxDiff = None

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'cat.jpg')
    img2_path = os.path.join(data_path, 'lena.jpg')
    img3_path = os.path.join(data_path, 'lena-face.jpg')

    def _run_face_area_filter(self,
                              dataset: Dataset,
                              target_list,
                              op,
                              num_proc=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=num_proc)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.remove_columns(Fields.stats)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_filter_small(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = FaceAreaFilter(min_ratio=0.4, max_ratio=1.0)
        self._run_face_area_filter(dataset, tgt_list, op)

    def test_filter_large(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = FaceAreaFilter(min_ratio=0.0, max_ratio=0.4)
        self._run_face_area_filter(dataset, tgt_list, op)

    def test_filter_multimodal(self):

        ds_list = [{
            'text': 'a test sentence',
            'images': []
        }, {
            'text': 'a test sentence',
            'images': [self.img1_path]
        }, {
            'text': 'a test sentence',
            'images': [self.img2_path]
        }, {
            'text': 'a test sentence',
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'text': 'a test sentence',
            'images': []
        }, {
            'text': 'a test sentence',
            'images': [self.img1_path]
        }, {
            'text': 'a test sentence',
            'images': [self.img2_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = FaceAreaFilter()
        self._run_face_area_filter(dataset, tgt_list, op)

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
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = FaceAreaFilter(min_ratio=0.0, max_ratio=0.4, any_or_all='any')
        self._run_face_area_filter(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = [{'images': [self.img1_path, self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = FaceAreaFilter(min_ratio=0.0, max_ratio=0.4, any_or_all='all')
        self._run_face_area_filter(dataset, tgt_list, op)

    def test_filter_multi_process(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = FaceAreaFilter()
        self._run_face_area_filter(dataset, tgt_list, op, num_proc=3)


if __name__ == '__main__':
    unittest.main()
