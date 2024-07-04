import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.video_aspect_ratio_filter import \
    VideoAspectRatioFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoAspectRatioFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # 640x360, 16:9
    vid2_path = os.path.join(data_path, 'video2.mp4')  # 480x640, 3:4
    vid3_path = os.path.join(data_path, 'video3.mp4')  # 362x640, 181:320

    def _run_op(self, dataset: Dataset, target_list, op, np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_default_params(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoAspectRatioFilter()
        self._run_op(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = VideoAspectRatioFilter(min_ratio='3/4',
                                    max_ratio='16/9',
                                    any_or_all='any')
        self._run_op(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid1_path, self.vid2_path]}]

        dataset = Dataset.from_list(ds_list)
        op = VideoAspectRatioFilter(min_ratio='3/4',
                                    max_ratio='16/9',
                                    any_or_all='all')
        self._run_op(dataset, tgt_list, op)

    def test_parallel(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid1_path]}, {'videos': [self.vid2_path]}]

        dataset = Dataset.from_list(ds_list)
        op = VideoAspectRatioFilter(min_ratio='3/4', max_ratio='16/9')
        self._run_op(dataset, tgt_list, op, np=2)


if __name__ == '__main__':
    unittest.main()
