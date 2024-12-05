import os
from data_juicer.core.dataset_builder import rewrite_cli_datapath
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, SKIPPED_TESTS

@SKIPPED_TESTS.register_module()
class DatasetBuilderTest(DataJuicerTestCaseBase):

    def test_rewrite_cli_datapath_local_single_file(self):
        dataset_path = "./data/sample.txt"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'path': ['./data/sample.txt'], 'type': 'local', 'weight': 1.0}], ans)

    def test_rewrite_cli_datapath_local_directory(self):
        dataset_path = "./data"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'path': ['./data'], 'type': 'local', 'weight': 1.0}], ans)

    def test_rewrite_cli_datapath_absolute_path(self):
        dataset_path = os.curdir + "/data/sample.txt"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'type': 'local', 'path': [dataset_path], 'weight': 1.0}], ans)

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
            [{'path': ['./data/sample.json'], 'type': 'local', 'weight': 0.5},
             {'path': ['./data/sample.txt'], 'type': 'local', 'weight': 1.0}],
            ans)