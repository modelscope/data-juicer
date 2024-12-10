import os
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO

from networkx.classes import is_empty

from data_juicer.config import init_configs
from data_juicer.core.data.dataset_builder import rewrite_cli_datapath
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, SKIPPED_TESTS

@SKIPPED_TESTS.register_module()
class DatasetBuilderTest(DataJuicerTestCaseBase):

    def test_rewrite_cli_datapath_local_single_file(self):
        dataset_path = "./data/sample.txt"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'path': ['./data/sample.txt'], 'type': 'ondisk', 'weight': 1.0}], ans)

    def test_rewrite_cli_datapath_local_directory(self):
        dataset_path = "./data"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'path': ['./data'], 'type': 'ondisk', 'weight': 1.0}], ans)

    def test_rewrite_cli_datapath_absolute_path(self):
        dataset_path = os.curdir + "/data/sample.txt"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'type': 'ondisk', 'path': [dataset_path], 'weight': 1.0}], ans)

    def test_rewrite_cli_datapath_hf(self):
        dataset_path = "hf-internal-testing/librispeech_asr_dummy"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual([{'path': 'hf-internal-testing/librispeech_asr_dummy',
                           'split': 'train',
                           'type': 'huggingface'}],
                         ans)

    def test_rewrite_cli_datapath_local_wrong_files(self):
        dataset_path = "./missingDir"
        self.assertRaisesRegex(ValueError, "Unable to load the dataset",
                               rewrite_cli_datapath, dataset_path)

    def test_rewrite_cli_datapath_with_weights(self):
        dataset_path = "0.5 ./data/sample.json ./data/sample.txt"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'path': ['./data/sample.json'], 'type': 'ondisk', 'weight': 0.5},
             {'path': ['./data/sample.txt'], 'type': 'ondisk', 'weight': 1.0}],
            ans)

    def test_dataset_builder_ondisk_config(self):
        test_config_file = './data/test_config.yaml'
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=f'--config {test_config_file}'.split())
            self.assertIsInstance(cfg, Namespace)
            self.assertEqual(cfg.project_name, 'dataset-ondisk-json')
            self.assertEqual(cfg.dataset,
                             {'path': ['sample.json'], 'type': 'ondisk'})
            self.assertEqual(not cfg.dataset_path, True)

    def test_dataset_builder_ondisk_config_list(self):
        test_config_file = './data/test_config_list.yaml'
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=f'--config {test_config_file}'.split())
            self.assertIsInstance(cfg, Namespace)
            self.assertEqual(cfg.project_name, 'dataset-ondisk-list')
            self.assertEqual(cfg.dataset,[
                {'path': ['sample.json'], 'type': 'ondisk'},
                {'path': ['sample.txt'], 'type': 'ondisk'}])
            self.assertEqual(not cfg.dataset_path, True)


if __name__ == '__main__':
    unittest.main()
