import os
import os.path as osp
import copy
import unittest
import json
import tempfile
import shutil
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_extract_frames_mapper import \
    VideoExtractFramesMapper
from data_juicer.utils.constant import Fields
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
        shutil.rmtree(self.tmp_dir)

    def _run_video_extract_frames_mapper(self,
                                            op,
                                            source_list,
                                            target_list,
                                            num_proc=1):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process, batch_size=2, num_proc=num_proc)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def _get_frames_list(self, filepath, frame_dir, frame_num):
        frames_dir = osp.join(frame_dir, osp.splitext(osp.basename(filepath))[0])
        frames_list = [osp.join(frames_dir, f'frame_{i}.jpg') for i in range(frame_num)]
        return frames_list

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

        tgt_list = copy.deepcopy(ds_list)
        tgt_list[0].update({Fields.video_frames: 
            json.dumps({self.vid1_path: self._get_frames_list(self.vid1_path, frame_dir, frame_num)})})
        tgt_list[1].update({Fields.video_frames: 
            json.dumps({self.vid2_path: self._get_frames_list(self.vid2_path, frame_dir, frame_num)})})
        tgt_list[2].update({Fields.video_frames: 
            json.dumps({self.vid3_path: self._get_frames_list(self.vid3_path, frame_dir, frame_num)})})
        
        op = VideoExtractFramesMapper(
            frame_sampling_method='uniform',
            frame_num=frame_num,
            frame_dir=frame_dir)
        self._run_video_extract_frames_mapper(op, ds_list, tgt_list)

    
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

        tgt_list = copy.deepcopy(ds_list)
        tgt_list[0].update({Fields.video_frames: 
            json.dumps({self.vid1_path: self._get_frames_list(self.vid1_path, frame_dir, 3)})})
        tgt_list[1].update({Fields.video_frames: json.dumps({
            self.vid2_path: self._get_frames_list(self.vid2_path, frame_dir, 3),
            self.vid3_path: self._get_frames_list(self.vid3_path, frame_dir, 6)
            })})
        tgt_list[2].update({Fields.video_frames: 
            json.dumps({self.vid3_path: self._get_frames_list(self.vid3_path, frame_dir, 6)})})
        
        op = VideoExtractFramesMapper(
            frame_sampling_method='all_keyframes',
            frame_dir=frame_dir)
        self._run_video_extract_frames_mapper(op, ds_list, tgt_list)



if __name__ == '__main__':
    unittest.main()
