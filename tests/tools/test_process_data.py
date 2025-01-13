import os
import os.path as osp
import shutil
import subprocess
import tempfile
import unittest
import uuid
import yaml

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


def run_in_subprocess(cmd):
    try:
        with subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE) as return_info:
            while True:
                next_line = return_info.stdout.readline()
                return_line = next_line.decode('utf-8', 'ignore').strip()
                if return_line == '' and return_info.poll() != None:
                    break
                if return_line != '':
                    print(return_line)

            err_lines = ''
            while True:
                next_line = return_info.stderr.readline()
                return_line = next_line.decode('utf-8', 'ignore').strip()
                if return_line == '' and return_info.poll() != None:
                    break
                if return_line != '':
                    print(return_line)
                    err_lines += return_line + '\n'

            return_code = return_info.wait()
            if return_code:
                raise RuntimeError(err_lines)
    except Exception as e:
        raise e


class ProcessDataTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.tmp_dir = tempfile.TemporaryDirectory().name
        os.makedirs(self.tmp_dir, exist_ok=True)

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


class ProcessDataRayTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = osp.join(cur_dir, f'tmp_{uuid.uuid4().hex}')
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()

        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        import ray
        ray.shutdown()

    def test_ray_image(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_0.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_0.json')
        text_keys = 'text'

        data_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
            'demos', 'data', 'demo-dataset-images.jsonl')
        yaml_config = {
            'dataset_path': data_path,
            'executor_type': 'ray',
            'ray_address': 'auto',
            'text_keys': text_keys,
            'image_key': 'images',
            'export_path': tmp_out_path,
            'process': [
                {
                    'image_nsfw_filter': {
                        'hf_nsfw_model': 'Falconsai/nsfw_image_detection',
                        'trust_remote_code': True,
                        'score_threshold': 0.5,
                        'any_or_all': 'any',
                        'mem_required': '8GB'
                    },
                    'image_aspect_ratio_filter':{
                        'min_ratio': 0.5,
                        'max_ratio': 2.0
                    }
                }
            ]
        }

        with open(tmp_yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        run_in_subprocess(f'python tools/process_data.py --config {tmp_yaml_file}')

        self.assertTrue(osp.exists(tmp_out_path))

        from datasets import load_dataset
        jsonl_files = [os.path.join(tmp_out_path, f) \
                       for f in os.listdir(tmp_out_path) \
                        if f.endswith('.json')]
        dataset = load_dataset(
            'json',
            data_files={'jsonl': jsonl_files})

        self.assertEqual(len(dataset['jsonl']), 3)
        for item in dataset['jsonl']:
            self.assertIn('aspect_ratios', item['__dj__stats__'])

    def test_ray_precise_dedup(self):
        tmp_yaml_file = osp.join(self.tmp_dir, 'config_1.yaml')
        tmp_out_path = osp.join(self.tmp_dir, 'output_dedup')
        text_keys = 'text'

        data_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
            'demos', 'data', 'demo-dataset-deduplication.jsonl')
        yaml_config = {
            'dataset_path': data_path,
            'executor_type': 'ray',
            'ray_address': 'auto',
            'text_keys': text_keys,
            'image_key': 'images',
            'export_path': tmp_out_path,
            'process': [
                {
                    'ray_document_deduplicator': {
                        'backend': 'ray_actor',
                    },
                }
            ]
        }

        with open(tmp_yaml_file, 'w') as file:
            yaml.dump(yaml_config, file)

        run_in_subprocess(f'python tools/process_data.py --config {tmp_yaml_file}')

        self.assertTrue(osp.exists(tmp_out_path))

        jsonl_files = [os.path.join(tmp_out_path, f) \
                       for f in os.listdir(tmp_out_path) \
                        if f.endswith('.json')]
        data_cnt = 0
        for file in jsonl_files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        data_cnt += 1

        self.assertEqual(data_cnt, 13)


if __name__ == '__main__':
    unittest.main()
