import os
import shutil
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.video_face_blur_mapper import VideoFaceBlurMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoFaceBlurMapperTest(DataJuicerTestCaseBase):

    maxDiff = None

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')
    vid5_path = os.path.join(data_path, 'video5.mp4')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.chk_path = os.path.join(cls.data_path, cls.__name__)
        shutil.rmtree(cls.chk_path, ignore_errors=True)
        os.makedirs(cls.chk_path)

    def _run_helper(self, op, source_list, np=1):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process, num_proc=np)
        res_list = dataset.to_list()
        for source, res in zip(source_list, res_list):
            self.assertEqual(len(source[op.video_key]), len(res[op.video_key]))
            # for manual check
            for path in res[op.video_key]:
                basename = os.path.basename(path)
                dst = f'{self.chk_path}/{op.blur_type}:{op.radius}_np:{np}_{basename}'
                shutil.copy(path, dst)

    def test_gaussian_radius(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid4_path]
        }, {
            'videos': [self.vid5_path]
        }]
        op = VideoFaceBlurMapper(blur_type='gaussian', radius=10)
        self._run_helper(op, ds_list)

    def test_box_radius(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid4_path]
        }, {
            'videos': [self.vid5_path]
        }]
        op = VideoFaceBlurMapper(blur_type='box', radius=10)
        self._run_helper(op, ds_list)

    def test_mean(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid4_path]
        }, {
            'videos': [self.vid5_path]
        }]
        op = VideoFaceBlurMapper(blur_type='mean')
        self._run_helper(op, ds_list)

    def test_gaussian_radius_parallel(self):
        import multiprocess as mp
        mp.set_start_method('forkserver', force=True)
    
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid4_path]
        }, {
            'videos': [self.vid5_path]
        }]
        op = VideoFaceBlurMapper(blur_type='gaussian', radius=10)
        self._run_helper(op, ds_list, np=3)

if __name__ == '__main__':
    unittest.main()