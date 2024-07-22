import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.audio_size_filter import AudioSizeFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AudioSizeFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    aud1_path = os.path.join(data_path, 'audio1.wav')  # 970574 / 948K
    aud2_path = os.path.join(data_path, 'audio2.wav')  # 2494872 / 2.4M
    aud3_path = os.path.join(data_path, 'audio3.ogg')  # 597254 / 583K

    def _run_audio_size_filter(self, dataset: Dataset, target_list, op, np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.audio_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_min_max(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = AudioSizeFilter(min_size='800kb', max_size='1MB')
        self._run_audio_size_filter(dataset, tgt_list, op)

    def test_min(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud1_path]}, {'audios': [self.aud2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = AudioSizeFilter(min_size='900kib')
        self._run_audio_size_filter(dataset, tgt_list, op)

    def test_max(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud1_path]}, {'audios': [self.aud3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = AudioSizeFilter(max_size='2MiB')
        self._run_audio_size_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'audios': [self.aud1_path, self.aud2_path]
        }, {
            'audios': [self.aud2_path, self.aud3_path]
        }, {
            'audios': [self.aud1_path, self.aud3_path]
        }]
        tgt_list = [{
            'audios': [self.aud1_path, self.aud2_path]
        }, {
            'audios': [self.aud1_path, self.aud3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = AudioSizeFilter(min_size='800kb',
                             max_size='1MB',
                             any_or_all='any')
        self._run_audio_size_filter(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'audios': [self.aud1_path, self.aud2_path]
        }, {
            'audios': [self.aud2_path, self.aud3_path]
        }, {
            'audios': [self.aud1_path, self.aud3_path]
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = AudioSizeFilter(min_size='800kb',
                             max_size='1MB',
                             any_or_all='all')
        self._run_audio_size_filter(dataset, tgt_list, op)

    def test_filter_in_parallel(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = AudioSizeFilter(min_size='800kb', max_size='1MB')
        self._run_audio_size_filter(dataset, tgt_list, op, np=2)


if __name__ == '__main__':
    unittest.main()
