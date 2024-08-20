import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.video_ffmpeg_wrapped_mapper import \
    VideoFFmpegWrappedMapper
from data_juicer.utils.mm_utils import close_video, load_video
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoFFmpegWrappedMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # 640x360, 16:9
    vid2_path = os.path.join(data_path, 'video2.mp4')  # 480x640, 3:4
    vid3_path = os.path.join(data_path, 'video3.mp4')  # 362x640, 181:320

    def _run_op(self, ds_list, target_list, op, np=1):
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=np)

        def get_size(dataset):
            sizes = []
            res_list = dataset.to_list()
            for sample in res_list:
                sample_list = []
                for value in sample['videos']:
                    video = load_video(value)
                    width = video.streams.video[0].codec_context.width
                    height = video.streams.video[0].codec_context.height
                    sample_list.append((width, height))
                    close_video(video)
                sizes.append(sample_list)
            return sizes

        sizes = get_size(dataset)
        self.assertEqual(sizes, target_list)

    def test_resize(self):
        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path, self.vid3_path]
        }]
        tgt_list = [[(400, 480), (400, 480), (400, 480)]]
        op = VideoFFmpegWrappedMapper('scale',
                                      filter_kwargs={
                                          'width': 400,
                                          'height': 480
                                      },
                                      capture_stderr=False)
        self._run_op(ds_list, tgt_list, op)

    def test_resize_parallel(self):
        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path, self.vid3_path]
        }]
        tgt_list = [[(400, 480), (400, 480), (400, 480)]]
        op = VideoFFmpegWrappedMapper('scale',
                                      filter_kwargs={
                                          'width': 400,
                                          'height': 480
                                      },
                                      capture_stderr=False)
        self._run_op(ds_list, tgt_list, op, np=2)


if __name__ == '__main__':
    unittest.main()
