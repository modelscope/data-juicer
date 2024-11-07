import os
import unittest

from datasets import Dataset

from data_juicer.ops.filter.video_motion_score_raft_filter import \
    VideoMotionScoreRaftFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoMotionScoreRaftFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # 10.766147
    vid2_path = os.path.join(data_path, 'video2.mp4')  # 10.098914
    vid3_path = os.path.join(data_path, 'video3.mp4')  # 2.0731936

    def _run_helper(self, op, source_list, target_list, np=1):
        dataset = Dataset.from_list(source_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_default(self):
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
        op = VideoMotionScoreRaftFilter()
        self._run_helper(op, ds_list, tgt_list)

    def test_downscale(self):
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
        }]
        op = VideoMotionScoreRaftFilter(min_score=1.0, size=128)
        self._run_helper(op, ds_list, tgt_list)

    def test_downscale_max(self):
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
        }]
        op = VideoMotionScoreRaftFilter(min_score=1.0, size=256, max_size=256)
        self._run_helper(op, ds_list, tgt_list)

    def test_downscale_relative(self):
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
        }]
        op = VideoMotionScoreRaftFilter(min_score=0.005, size=(128, 160), relative=True)
        self._run_helper(op, ds_list, tgt_list)

    def test_high(self):
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
        }]
        op = VideoMotionScoreRaftFilter(min_score=10)
        self._run_helper(op, ds_list, tgt_list)

    def test_low(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid3_path]}]
        op = VideoMotionScoreRaftFilter(min_score=0.0, max_score=3)
        self._run_helper(op, ds_list, tgt_list)

    def test_middle(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        op = VideoMotionScoreRaftFilter(min_score=3, max_score=10.5)
        self._run_helper(op, ds_list, tgt_list)

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
        op = VideoMotionScoreRaftFilter(min_score=3,
                                    max_score=10.5,
                                    any_or_all='any')
        self._run_helper(op, ds_list, tgt_list)

    def test_all(self):
        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = []
        op = VideoMotionScoreRaftFilter(min_score=3,
                                    max_score=10.5,
                                    any_or_all='all')
        self._run_helper(op, ds_list, tgt_list)

    def test_parallel(self):
        import multiprocess as mp
        mp.set_start_method('spawn', force=True)

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        op = VideoMotionScoreRaftFilter(min_score=3, max_score=10.5)
        self._run_helper(op, ds_list, tgt_list, np=2)


if __name__ == '__main__':
    unittest.main()
