# flake8: noqa: E501

import os
import unittest

from data_juicer.core.data import NestedDataset
from data_juicer.ops.mapper.video_split_by_key_frame_mapper import \
    VideoSplitByKeyFrameMapper
from data_juicer.utils.file_utils import add_suffix_to_filename
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoSplitByKeyFrameMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')

    def _get_res_list(self, dataset, source_list):
        res_list = []
        origin_paths = [self.vid1_path, self.vid2_path, self.vid3_path]
        idx = 0
        for sample in dataset.to_list():
            output_paths = sample['videos']

            # for keep_original_sample=True
            if set(output_paths) <= set(origin_paths):
                res_list.append(sample)
                continue

            source = source_list[idx]
            idx += 1

            output_file_names = [
                os.path.splitext(os.path.basename(p))[0] for p in output_paths
            ]
            split_frames_nums = []
            for origin_path in source['videos']:
                origin_file_name = os.path.splitext(
                    os.path.basename(origin_path))[0]
                cnt = 0
                for output_file_name in output_file_names:
                    if origin_file_name in output_file_name:
                        cnt += 1
                split_frames_nums.append(cnt)

            res_list.append({
                'text': sample['text'],
                'split_frames_num': split_frames_nums
            })

        return res_list

    def _run_video_split_by_key_frame_mapper(self,
                                             op,
                                             source_list,
                                             target_list,
                                             num_proc=1):
        dataset = NestedDataset.from_list(source_list)
        dataset = dataset.map(op.process, num_proc=num_proc)
        res_list = self._get_res_list(dataset, source_list)
        self.assertEqual(res_list, target_list)

    def test(self):
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
        tgt_list = [{
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [6]
        }]
        op = VideoSplitByKeyFrameMapper(keep_original_sample=False)
        self._run_video_split_by_key_frame_mapper(op, ds_list, tgt_list)

    def test_keep_ori_sample(self):
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
        tgt_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [6]
        }]
        op = VideoSplitByKeyFrameMapper()
        self._run_video_split_by_key_frame_mapper(op, ds_list, tgt_list)

    def test_multi_process(self):
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
        tgt_list = [{
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [6]
        }]
        op = VideoSplitByKeyFrameMapper(keep_original_sample=False)
        self._run_video_split_by_key_frame_mapper(op,
                                                  ds_list,
                                                  tgt_list,
                                                  num_proc=2)

    def test_multi_chunk(self):
        ds_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3, 3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [3, 6]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [3, 6]
        }]
        op = VideoSplitByKeyFrameMapper(keep_original_sample=False)
        self._run_video_split_by_key_frame_mapper(op, ds_list, tgt_list)


if __name__ == '__main__':
    unittest.main()
