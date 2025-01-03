import os
import os.path as osp
import re
import copy
import unittest
import json
import tempfile
import shutil
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_extract_frames_mapper import \
    VideoExtractFramesMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoExtractFramesMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    tmp_dir = tempfile.TemporaryDirectory().name

    def tearDown(self):
        super().tearDown()
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        default_frame_dir_prefix = self._get_default_frame_dir_prefix()
        if osp.exists(default_frame_dir_prefix):
            shutil.rmtree(osp.dirname(default_frame_dir_prefix))

    def _get_default_frame_dir_prefix(self):
        from data_juicer.ops.mapper.video_extract_frames_mapper import OP_NAME
        default_frame_dir_prefix = osp.abspath(osp.join(self.data_path, 
            f'{Fields.multimodal_data_output_dir}/{OP_NAME}/'))
        return default_frame_dir_prefix

    def _get_frames_list(self, filepath, frame_dir, frame_num):
        frames_dir = osp.join(frame_dir, osp.splitext(osp.basename(filepath))[0])
        frames_list = [osp.join(frames_dir, f'frame_{i}.jpg') for i in range(frame_num)]
        return frames_list

    def _get_frames_dir(self, filepath, frame_dir):
        frames_dir = osp.join(frame_dir, osp.splitext(osp.basename(filepath))[0])
        return frames_dir

    def _sort_files(self, file_list):
        return sorted(file_list, key=lambda x: int(re.search(r'(\d+)', x).group()))

    def test_duration(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]

        frame_num = 2
        frame_dir=os.path.join(self.tmp_dir, 'test1')
        vid1_frame_dir =  self._get_frames_dir(self.vid1_path, frame_dir)
        vid2_frame_dir =  self._get_frames_dir(self.vid2_path, frame_dir)
        vid3_frame_dir =  self._get_frames_dir(self.vid3_path, frame_dir)

        tgt_list = copy.deepcopy(ds_list)
        tgt_list[0].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid1_path: vid1_frame_dir})}})
        tgt_list[1].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid2_path: vid2_frame_dir})}})
        tgt_list[2].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid3_path: vid3_frame_dir})}})

        op = VideoExtractFramesMapper(
            frame_sampling_method='uniform',
            frame_num=frame_num,
            duration=0,
            frame_dir=frame_dir,
            batch_size=2,
            num_proc=1)

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()
        self.assertEqual(res_list, tgt_list)
        self.assertListEqual(
            self._sort_files(os.listdir(vid1_frame_dir)),
            [f'frame_{i}.jpg' for i in range(frame_num)])
        self.assertListEqual(
            self._sort_files(os.listdir(vid2_frame_dir)),
            [f'frame_{i}.jpg' for i in range(frame_num)])
        self.assertListEqual(
            self._sort_files(os.listdir(vid3_frame_dir)),
            [f'frame_{i}.jpg' for i in range(frame_num)])

    def test_uniform_sampling(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        frame_num = 3
        frame_dir=os.path.join(self.tmp_dir, 'test1')
        vid1_frame_dir =  self._get_frames_dir(self.vid1_path, frame_dir)
        vid2_frame_dir =  self._get_frames_dir(self.vid2_path, frame_dir)
        vid3_frame_dir =  self._get_frames_dir(self.vid3_path, frame_dir)

        tgt_list = copy.deepcopy(ds_list)
        tgt_list[0].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid1_path: vid1_frame_dir})}})
        tgt_list[1].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid2_path: vid2_frame_dir})}})
        tgt_list[2].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid3_path: vid3_frame_dir})}})

        op = VideoExtractFramesMapper(
            frame_sampling_method='uniform',
            frame_num=frame_num,
            duration=10,
            frame_dir=frame_dir,
            batch_size=2,
            num_proc=1)

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()
        self.assertEqual(res_list, tgt_list)
        self.assertListEqual(
            self._sort_files(os.listdir(vid1_frame_dir)),
            [f'frame_{i}.jpg' for i in range(3)])
        self.assertListEqual(
            self._sort_files(os.listdir(vid2_frame_dir)),
            [f'frame_{i}.jpg' for i in range(6)])
        self.assertListEqual(
            self._sort_files(os.listdir(vid3_frame_dir)),
            [f'frame_{i}.jpg' for i in range(12)])

    def test_all_keyframes_sampling(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}' + \
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        frame_dir=os.path.join(self.tmp_dir, 'test2')
        vid1_frame_dir =  self._get_frames_dir(self.vid1_path, frame_dir)
        vid2_frame_dir =  self._get_frames_dir(self.vid2_path, frame_dir)
        vid3_frame_dir =  self._get_frames_dir(self.vid3_path, frame_dir)

        tgt_list = copy.deepcopy(ds_list)
        tgt_list[0].update({Fields.meta: {MetaKeys.video_frames: 
            json.dumps({self.vid1_path: vid1_frame_dir})}})
        tgt_list[1].update({Fields.meta: {MetaKeys.video_frames: json.dumps({
            self.vid2_path: vid2_frame_dir,
            self.vid3_path: vid3_frame_dir
            })}})
        tgt_list[2].update({Fields.meta: {MetaKeys.video_frames: 
            json.dumps({self.vid3_path: vid3_frame_dir})}})
        
        op = VideoExtractFramesMapper(
            frame_sampling_method='all_keyframes',
            frame_dir=frame_dir,
            duration=5,
            batch_size=2,
            num_proc=2)

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()
        self.assertEqual(res_list, tgt_list)
        self.assertListEqual(
            self._sort_files(os.listdir(vid1_frame_dir)),
            [f'frame_{i}.jpg' for i in range(4)])
        self.assertListEqual(
            self._sort_files(os.listdir(vid2_frame_dir)),
            [f'frame_{i}.jpg' for i in range(5)])
        self.assertListEqual(
            self._sort_files(os.listdir(vid3_frame_dir)),
            [f'frame_{i}.jpg' for i in range(13)])

    def test_default_frame_dir(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]

        frame_num = 2
        op = VideoExtractFramesMapper(
            frame_sampling_method='uniform',
            frame_num=frame_num,
            duration=5,
            batch_size=2,
            num_proc=1
            )

        vid1_frame_dir =  op._get_default_frame_dir(self.vid1_path)
        vid2_frame_dir =  op._get_default_frame_dir(self.vid2_path)
        vid3_frame_dir =  op._get_default_frame_dir(self.vid3_path)

        tgt_list = copy.deepcopy(ds_list)
        tgt_list[0].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid1_path: vid1_frame_dir})}})
        tgt_list[1].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid2_path: vid2_frame_dir})}})
        tgt_list[2].update({Fields.meta: {MetaKeys.video_frames: json.dumps({self.vid3_path: vid3_frame_dir})}})

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()

        frame_dir_prefix = self._get_default_frame_dir_prefix()
        self.assertIn(frame_dir_prefix, osp.abspath(vid1_frame_dir))
        self.assertIn(frame_dir_prefix, osp.abspath(vid2_frame_dir))
        self.assertIn(frame_dir_prefix, osp.abspath(vid3_frame_dir))

        self.assertEqual(res_list, tgt_list)

        self.assertListEqual(
            self._sort_files(os.listdir(vid1_frame_dir)),
            [f'frame_{i}.jpg' for i in range(4)])
        self.assertListEqual(
            self._sort_files(os.listdir(vid2_frame_dir)),
            [f'frame_{i}.jpg' for i in range(8)])
        self.assertListEqual(
            self._sort_files(os.listdir(vid3_frame_dir)),
            [f'frame_{i}.jpg' for i in range(18)])


if __name__ == '__main__':
    unittest.main()
