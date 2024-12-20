import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.audio_duration_filter import AudioDurationFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG

class AudioDurationFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    aud1_path = os.path.join(data_path, 'audio1.wav')  # about 6s
    aud2_path = os.path.join(data_path, 'audio2.wav')  # about 14s
    aud3_path = os.path.join(data_path, 'audio3.ogg')  # about 1min59s

    def _run_audio_duration_filter(self,
                                   dataset: Dataset,
                                   target_list,
                                   op,
                                   np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.audio_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    @TEST_TAG("standalone", "ray")
    def test_default_filter(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        dataset = self.generate_dataset(ds_list)
        op = AudioDurationFilter()
        result = self.run_single_op(dataset, op, [op.audio_key])
        self.assertDatasetEqual(result, tgt_list)


    @TEST_TAG("standalone", "ray")
    def test_filter_long_audios(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud1_path]}]
        dataset = self.generate_dataset(ds_list)
        op = AudioDurationFilter(max_duration=10)
        result = self.run_single_op(dataset, op, [op.audio_key])
        self.assertDatasetEqual(result, tgt_list)

    @TEST_TAG("standalone", "ray")
    def test_filter_short_audios(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud3_path]}]
        dataset = self.generate_dataset(ds_list)
        op = AudioDurationFilter(min_duration=60)
        result = self.run_single_op(dataset, op, [op.audio_key])
        self.assertDatasetEqual(result, tgt_list)

    @TEST_TAG("standalone", "ray")
    def test_filter_audios_within_range(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud2_path]}]
        dataset = self.generate_dataset(ds_list)
        op = AudioDurationFilter(min_duration=10, max_duration=20)
        result = self.run_single_op(dataset, op, [op.audio_key])
        self.assertDatasetEqual(result, tgt_list)

    @TEST_TAG("standalone", "ray")
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
            'audios': [self.aud2_path, self.aud3_path]
        }]
        dataset = self.generate_dataset(ds_list)
        op = AudioDurationFilter(min_duration=10,
                                 max_duration=20,
                                 any_or_all='any')
        result = self.run_single_op(dataset, op, [op.audio_key])
        self.assertDatasetEqual(result, tgt_list)

    @TEST_TAG("standalone", "ray")
    def test_all(self):

        ds_list = [{
            'audios': [self.aud1_path, self.aud2_path]
        }, {
            'audios': [self.aud2_path, self.aud3_path]
        }, {
            'audios': [self.aud1_path, self.aud3_path]
        }]
        tgt_list = []
        dataset = self.generate_dataset(ds_list)
        op = AudioDurationFilter(min_duration=10,
                                 max_duration=20,
                                 any_or_all='all')
        result = self.run_single_op(dataset, op, [op.audio_key])
        self.assertDatasetEqual(result, tgt_list)

    @TEST_TAG("standalone", "ray")
    def test_filter_in_parallel(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud2_path]}]
        dataset = self.generate_dataset(ds_list)
        op = AudioDurationFilter(min_duration=10, max_duration=20)
        result = self.run_single_op(dataset, op, [op.audio_key])
        self.assertDatasetEqual(result, tgt_list)


if __name__ == '__main__':
    unittest.main()
