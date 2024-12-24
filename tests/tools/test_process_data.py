import os
import os.path as osp
import shutil
import subprocess
import tempfile
import unittest
import yaml

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


def run_in_subprocess(cmd):
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"Standard Output: {result.stdout}")
        print(f"Standard Error: {result.stderr}")
        raise subprocess.CalledProcessError(result, cmd)

    return result


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


class ProcessDataRayTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        # self._auto_create_ray_cluster()
        self.tmp_dir = f'/workspace/tmp/{self.__class__.__name__}'
        if not osp.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def _auto_create_ray_cluster(self):
        try:
            # ray cluster already exists, return
            run_in_subprocess('ray status')
            self.tmp_ray_cluster = False
            return
        except:
            pass

        self.tmp_ray_cluster = True
        head_port = '6379'
        head_addr = '127.0.0.1'
        rank = int(os.environ.get('RANK', 0))

        if rank == 0:
            cmd = f"ray start --head --port={head_port} --node-ip-address={head_addr}"
        else:
            cmd = f"ray start --address={head_addr}:{head_port}"

        print(f"current rank: {rank}; execute cmd: {cmd}")

        run_in_subprocess(cmd)
        
    def _close_ray_cluster(self):
        run_in_subprocess('ray stop')

    def tearDown(self):
        super().tearDown()

        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        import ray
        ray.shutdown()

        # if self.tmp_ray_cluster:
        #     self._close_ray_cluster()

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

        import ray
        res_ds = ray.data.read_json(tmp_out_path)
        res_ds = res_ds.to_pandas().to_dict(orient='records')

        self.assertEqual(len(res_ds), 3)
        for item in res_ds:
            self.assertIn('aspect_ratios', item['__dj__stats__'])


if __name__ == '__main__':
    unittest.main()
