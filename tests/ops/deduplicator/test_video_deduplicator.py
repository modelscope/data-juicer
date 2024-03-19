import os
import unittest

from datasets import Dataset

from data_juicer.ops.deduplicator.video_deduplicator import VideoDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoDeduplicatorTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    video1_path = os.path.join(data_path, 'video1.mp4')
    video2_path = os.path.join(data_path, 'video2.mp4')
    video3_path = os.path.join(data_path, 'video3.mp4')
    # video1_dup.mp4 is a duplicate sample of video1.mp4
    video4_path = os.path.join(data_path, 'video1_dup.mp4')
    if not os.path.exists(video4_path):
        os.symlink(video1_path, video4_path)
    # video2_dup.mp4 is a duplicate sample of video2.mp4
    video5_path = os.path.join(data_path, 'video2_dup.mp4')
    if not os.path.exists(video5_path):
        os.symlink(video2_path, video5_path)
    # video3_dup.mp4 is a duplicate sample of video3.mp4
    video6_path = os.path.join(data_path, 'video3_dup.mp4')
    if not os.path.exists(video6_path):
        os.symlink(video3_path, video6_path)
    # video3_dup_dup.mp4 is a duplicate sample of video6.mp4
    video7_path = os.path.join(data_path, 'video3_dup_dup.mp4')
    if not os.path.exists(video7_path):
        os.symlink(video6_path, video7_path)

    def _run_video_deduplicator(self, dataset: Dataset, target_list, op):

        dataset = dataset.map(op.compute_hash)
        dataset, _ = op.process(dataset)
        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_1(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    def test_2(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video2_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    def test_3(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }, {
            'videos': [self.video4_path]
        }, {
            'videos': [self.video5_path]
        }, {
            'videos': [self.video6_path]
        }, {
            'videos': [self.video7_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    def test_4(self):

        ds_list = [{
            'videos': [self.video1_path, self.video2_path, self.video3_path]
        }, {
            'videos': [self.video4_path, self.video5_path, self.video6_path]
        }, {
            'videos': [self.video7_path, self.video5_path]
        }, {
            'videos': [self.video6_path, self.video5_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path, self.video2_path, self.video3_path]
        }, {
            'videos': [self.video7_path, self.video5_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    def test_5(self):

        ds_list = [{
            'videos': [self.video1_path, self.video2_path]
        }, {
            'videos': [self.video2_path, self.video1_path]
        }, {
            'videos': [self.video4_path, self.video5_path]
        }, {
            'videos': [self.video7_path, self.video7_path]
        }, {
            'videos': [self.video6_path, self.video6_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path, self.video2_path]
        }, {
            'videos': [self.video2_path, self.video1_path]
        }, {
            'videos': [self.video7_path, self.video7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
