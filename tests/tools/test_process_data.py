import os
import os.path as osp
import shutil
import subprocess
import tempfile
import unittest
import yaml

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ProcessDataTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not osp.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def _test_status_code(self, yaml_file, output_path, text_keys):
        data_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
            'demos', 'data', 'demo-dataset.jsonl')
        yaml_config = {
            'dataset_path': data_path,
            'text_keys': text_keys,
            'np': 2,
            'export_path': output_path,
            'process': [
                {
                    'clean_copyright_mapper': None
                }
            ]
        }

        with open(yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        status_code = subprocess.call(
            f'python tools/process_data.py --config {yaml_file}', shell=True)

        return status_code

    def test_status_code_0(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_0.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_0.json')
        text_keys = 'text'

        status_code = self._test_status_code(tmp_yaml_file, tmp_out_path, text_keys)

        self.assertEqual(status_code, 0)
        self.assertTrue(osp.exists(tmp_out_path))

    def test_status_code_1(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_1.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_1.json')
        text_keys = 'keys_not_exists'

        status_code = self._test_status_code(tmp_yaml_file, tmp_out_path, text_keys)

        self.assertEqual(status_code, 1)
        self.assertFalse(osp.exists(tmp_out_path))


if __name__ == '__main__':
    unittest.main()
