import os
import multiprocess as mp
import unittest
from unittest.mock import patch, MagicMock

import torch
import ray

from data_juicer.utils.process_utils import setup_mp, get_min_cuda_memory, calculate_np
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG
from data_juicer.utils.constant import RAY_JOB_ENV_VAR

class ProcessUtilsTest(DataJuicerTestCaseBase):

    def test_setup_mp(self):
        all_methods = mp.get_all_start_methods()
        setup_mp()
        self.assertIn(mp.get_start_method(), all_methods)

        setup_mp('spawn')
        self.assertEqual(mp.get_start_method(), 'spawn')

        setup_mp(['spawn', 'forkserver', 'fork'])
        self.assertEqual(mp.get_start_method(), 'spawn')

    def test_get_min_cuda_memory(self):
        if torch.cuda.is_available():
            self.assertIsInstance(get_min_cuda_memory(), int)
        else:
            with self.assertRaises(AssertionError):
                get_min_cuda_memory()


class CalculateNpTest(DataJuicerTestCaseBase):

    def setUp(self):
        self._patch_module = 'data_juicer.utils.process_utils'
        self._patch_ray_module = 'data_juicer.utils.ray_utils'
        self._ori_ray_job_env_value = os.environ.get(RAY_JOB_ENV_VAR, '0')
        super().setUp()
    
    def tearDown(self):
        os.environ[RAY_JOB_ENV_VAR] = self._ori_ray_job_env_value
        return super().tearDown()

    def enable_ray_mode(self):
        os.environ[RAY_JOB_ENV_VAR] = '1'
        ray.init(address='auto', ignore_reinit_error=True)

    @TEST_TAG('ray')
    def test_cuda_mem_required_zero_and_num_proc_not_given(self):
        logger = MagicMock()
        with patch(f"{self._patch_ray_module}.get_ray_nodes_info") as mock_ray_nodes_info, \
            patch(f"{self._patch_module}.cuda_device_count") as mock_cuda_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_ray_nodes_info.return_value = {
                'node1_id': {'free_memory': 512, 'cpu_count': 8, 'free_gpus_memory': [2 * 1024]},
                'node2_id': {'free_memory': 512, 'cpu_count': 8, 'free_gpus_memory': [2 * 1024]},
                }
            mock_cuda_count.return_value = 2
            self.enable_ray_mode()
            result = calculate_np("test_op", mem_required=0, cpu_required=0, use_cuda=True)
            self.assertEqual(result, 2)
            logger.warning.assert_called_with(
                "The required cuda memory and gpu of Op[test_op] has not been specified. "
                "Please specify the mem_required field or gpu_required field in the config file. "
                "You can reference the config_all.yaml file.Set the auto `num_proc` to number of GPUs 2."
            )
    @TEST_TAG('ray')
    def test_cuda_auto_less_than_device_count(self):
        logger = MagicMock()
        with patch(f"{self._patch_ray_module}.get_ray_nodes_info") as mock_ray_nodes_info, \
            patch(f"{self._patch_module}.logger", logger):
            mock_ray_nodes_info.return_value = {
                'node1_id': {'free_memory': 512, 'cpu_count': 8, 'free_gpus_memory': [2 * 1024]},
                'node2_id': {'free_memory': 512, 'cpu_count': 8, 'free_gpus_memory': [2 * 1024]},
                }
            self.enable_ray_mode()
            result = calculate_np("test_op", mem_required=3, cpu_required=0, use_cuda=True)
            self.assertEqual(result, 2)
            logger.info.assert_called_with(
                "Set the auto `num_proc` to 2 of Op[test_op] based on the required cuda memory: 3GB required gpu: 0 and required cpu: 0."
            )

    @TEST_TAG('ray')
    def test_cuda_num_proc_exceeds_auto(self):
        logger = MagicMock()
        with patch(f"{self._patch_module}.available_gpu_memories") as mock_avail_gpu, \
            patch(f"{self._patch_module}.cuda_device_count") as mock_cuda_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_avail_gpu.return_value = [5 * 1024, 5 * 1024]  # 5GB per GPU
            mock_cuda_count.return_value = 2
            # auto_num_proc = (5//2) * 2 = 4
            self.enable_ray_mode()
            result = calculate_np("test_op", mem_required=2, cpu_required=0, use_cuda=True)
            self.assertEqual(result, 4)
            logger.info.assert_called_with(
                "Set the auto `num_proc` to 4 of Op[test_op] based on the required cuda memory: 2GB required gpu: 0 and required cpu: 0."
            )

    def test_cpu_default_num_proc(self):
        logger = MagicMock()
        with patch(f"{self._patch_module}.available_memories") as mock_avail_mem, \
            patch(f"{self._patch_module}.cpu_count") as mock_cpu_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_avail_mem.return_value = [8 * 1024 + 1]  # 8GB, add eps
            mock_cpu_count.return_value = 4
            result = calculate_np("test_op", mem_required=2, cpu_required=0, use_cuda=False)
            # auto_proc = 8//2 =4
            self.assertEqual(result, 4)
            logger.info.assert_called_with(
                "Set the auto `num_proc` to 4 of Op[test_op] based on the required memory: 2GB and required cpu: 0."
            )

    def test_cpu_insufficient_memory(self):
        logger = MagicMock()
        with patch(f"{self._patch_module}.available_memories") as mock_avail_mem, \
            patch(f"{self._patch_module}.cpu_count") as mock_cpu_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_avail_mem.return_value = [2 * 1024]  # 2GB
            mock_cpu_count.return_value = 8
            result = calculate_np("test_op", mem_required=3, cpu_required=2, use_cuda=False)
            # auto_proc = 0,  max(min(5,0),1) =1
            self.assertEqual(result, 1)
            logger.warning.assert_called_with(
                "The required CPU number: 2 "
                "and memory: 3GB might "
                "be more than the available CPU: 8 "
                "and memory: [2.0]GB."
                "This Op [test_op] might "
                "require more resource to run. "
                "Set the auto `num_proc` to available nodes number 1."
            )

    def test_cpu_num_proc_unset_and_mem_unlimited(self):
        logger = MagicMock()
        with patch(f"{self._patch_module}.available_memories") as mock_avail_mem, \
            patch(f"{self._patch_module}.cpu_count") as mock_cpu_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_avail_mem.return_value = [8 * 1024]
            mock_cpu_count.return_value = 4
            result = calculate_np("test_op", mem_required=0, cpu_required=0, use_cuda=False)
            # auto_proc = 8/(接近0) ≈无限大，取默认 num_proc=4
            self.assertEqual(result, 4)
            logger.info.assert_called_with(
                "Set the auto `num_proc` to 4 of Op[test_op] based on the required memory: 0GB and required cpu: 0."
            )


if __name__ == '__main__':
    unittest.main()
