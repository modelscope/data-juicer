import os
import unittest

import librosa
from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.audio_ffmpeg_wrapped_mapper import \
    AudioFFmpegWrappedMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AudioFFmpegWrappedMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    aud1_path = os.path.join(data_path, 'audio1.wav')  # 5.501678004535147
    aud2_path = os.path.join(data_path, 'audio2.wav')  # 14.142426303854876
    aud3_path = os.path.join(data_path, 'audio3.ogg')  # 119.87591836734694

    def _run_op(self, ds_list, target_list, op, np=1):
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=np)

        def get_size(dataset):
            durations = []
            res_list = dataset.to_list()
            for sample in res_list:
                sample_durations = []
                for aud_path in sample['audios']:
                    sample_durations.append(
                        librosa.get_duration(path=aud_path))
                durations.append(sample_durations)
            return durations

        sizes = get_size(dataset)
        self.assertEqual(sizes, target_list)

    def test_resize(self):
        ds_list = [{
            'audios': [self.aud1_path, self.aud2_path, self.aud3_path]
        }]
        tgt_list = [[5.501678004535147, 6.0, 6.0]]
        op = AudioFFmpegWrappedMapper('atrim',
                                      filter_kwargs={'end': 6},
                                      capture_stderr=False)
        self._run_op(ds_list, tgt_list, op)

    def test_resize_parallel(self):
        ds_list = [{
            'audios': [self.aud1_path, self.aud2_path, self.aud3_path]
        }]
        tgt_list = [[5.501678004535147, 6.0, 6.0]]
        op = AudioFFmpegWrappedMapper('atrim',
                                      filter_kwargs={'end': 6},
                                      capture_stderr=False)
        self._run_op(ds_list, tgt_list, op, np=2)


if __name__ == '__main__':
    unittest.main()
