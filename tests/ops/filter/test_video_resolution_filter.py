import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.video_resolution_filter import \
    VideoResolutionFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoResolutionFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    # video1: horizontal resolution 640p, vertical resolution 360p
    # video2: horizontal resolution 480p, vertical resolution 640p
    # video3: horizontal resolution 362p, vertical resolution 640p
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')

    def _run_video_resolution_filter(self,
                                     dataset: Dataset,
                                     target_list,
                                     op,
                                     np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_default_filter(self):

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
        op = VideoResolutionFilter()
        self._run_video_resolution_filter(dataset, tgt_list, op)

    def test_filter_low_resolution_videos(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoResolutionFilter(min_width=480, min_height=480)
        self._run_video_resolution_filter(dataset, tgt_list, op)

    def test_filter_high_resolution_videos(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoResolutionFilter(max_width=640, max_height=480)
        self._run_video_resolution_filter(dataset, tgt_list, op)

    def test_filter_videos_within_range(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoResolutionFilter(min_width=400, max_width=500)
        self._run_video_resolution_filter(dataset, tgt_list, op)

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
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoResolutionFilter(min_width=400,
                                   max_width=500,
                                   any_or_all='any')
        self._run_video_resolution_filter(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = VideoResolutionFilter(min_width=400,
                                   max_width=500,
                                   any_or_all='all')
        self._run_video_resolution_filter(dataset, tgt_list, op)

    def test_filter_in_parallel(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoResolutionFilter(min_width=400, max_width=500)
        self._run_video_resolution_filter(dataset, tgt_list, op, np=2)


if __name__ == '__main__':
    unittest.main()
