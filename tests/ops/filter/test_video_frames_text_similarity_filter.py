import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.video_frames_text_similarity_filter import \
    VideoFramesTextSimilarityFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoFramesTextSimilarityFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    # vid1: keyframes -- 0.2515, uniform-2 -- 0.2378, uniform-3 -- 0.2342
    # vid2: keyframes -- 0.2686, uniform-2 -- 0.2741, uniform-3 -- 0.2697
    # vid3: keyframes -- 0.3020, uniform-2 -- 0.3044, uniform-3 -- 0.2998
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')

    hf_clip = 'openai/clip-vit-base-patch32'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_clip)

    def _run_video_frames_text_similarity_filter(self,
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
            'videos': [self.vid1_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
        }, {
            'videos': [self.vid2_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid3_path],
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(self.hf_clip)
        self._run_video_frames_text_similarity_filter(dataset, tgt_list, op)

    def test_filter_large_score_videos(self):
        ds_list = [{
            'videos': [self.vid1_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
        }, {
            'videos': [self.vid2_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid3_path],
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = [{'videos': [self.vid1_path]}, {'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(self.hf_clip, max_score=0.3)
        self._run_video_frames_text_similarity_filter(dataset, tgt_list, op)

    def test_filter_small_score_videos(self):
        ds_list = [{
            'videos': [self.vid1_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
        }, {
            'videos': [self.vid2_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid3_path],
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = [{'videos': [self.vid2_path]}, {'videos': [self.vid3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(self.hf_clip, min_score=0.26)
        self._run_video_frames_text_similarity_filter(dataset, tgt_list, op)

    def test_filter_videos_within_range_keyframes(self):
        ds_list = [{
            'videos': [self.vid1_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
        }, {
            'videos': [self.vid2_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid3_path],
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(self.hf_clip,
                                             min_score=0.26,
                                             max_score=0.3)
        self._run_video_frames_text_similarity_filter(dataset, tgt_list, op)

    def test_filter_uniform_frames(self):
        ds_list = [{
            'videos': [self.vid1_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
        }, {
            'videos': [self.vid2_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid3_path],
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = [{'videos': [self.vid2_path]}, {'videos': [self.vid3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(self.hf_clip,
                                             min_score=0.26,
                                             max_score=0.3,
                                             frame_sampling_method='uniform')
        self._run_video_frames_text_similarity_filter(dataset, tgt_list, op)

    def test_filter_uniform_frames_with_different_frame_num(self):
        ds_list = [{
            'videos': [self.vid1_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
        }, {
            'videos': [self.vid2_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid3_path],
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(self.hf_clip,
                                             min_score=0.26,
                                             max_score=0.3,
                                             frame_sampling_method='uniform',
                                             frame_num=2)
        self._run_video_frames_text_similarity_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。'
            f'{SpecialTokens.eoc} {SpecialTokens.video} 身穿白色上衣的男子，'
            f'拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid2_path, self.vid3_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc} 两个长头发的女子正坐在一张圆桌前讲话互动。 '
            f'{SpecialTokens.video} {SpecialTokens.eoc}',
        }, {
            'videos': [self.vid1_path, self.vid3_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。'
            f'{SpecialTokens.eoc} 两个长头发的女子正坐在一张圆桌前讲话互动。 '
            f'{SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(self.hf_clip,
                                             min_score=0.26,
                                             max_score=0.3,
                                             frame_sampling_method='uniform',
                                             frame_num=2,
                                             any_or_all='any')
        self._run_video_frames_text_similarity_filter(dataset, tgt_list, op)

    def test_all(self):
        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。'
            f'{SpecialTokens.eoc} {SpecialTokens.video} 身穿白色上衣的男子，'
            f'拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid2_path, self.vid3_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc} 两个长头发的女子正坐在一张圆桌前讲话互动。 '
            f'{SpecialTokens.video} {SpecialTokens.eoc}',
        }, {
            'videos': [self.vid1_path, self.vid3_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。'
            f'{SpecialTokens.eoc} 两个长头发的女子正坐在一张圆桌前讲话互动。 '
            f'{SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(self.hf_clip,
                                             min_score=0.26,
                                             max_score=0.3,
                                             frame_sampling_method='uniform',
                                             frame_num=2,
                                             any_or_all='all')
        self._run_video_frames_text_similarity_filter(dataset, tgt_list, op)

    def test_filter_in_parallel(self):

        ds_list = [{
            'videos': [self.vid1_path],
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
        }, {
            'videos': [self.vid2_path],
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
        }, {
            'videos': [self.vid3_path],
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoFramesTextSimilarityFilter(
            self.hf_clip,
            min_score=0.26,
            max_score=0.3,
        )
        self._run_video_frames_text_similarity_filter(dataset,
                                                      tgt_list,
                                                      op,
                                                      np=2)


if __name__ == '__main__':
    unittest.main()
