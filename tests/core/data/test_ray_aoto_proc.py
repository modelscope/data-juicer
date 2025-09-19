import unittest
from unittest.mock import MagicMock, patch
from data_juicer.core.data.ray_dataset import RayDataset


class TestRayDataset(unittest.TestCase):
    def setUp(self):
        self.mock_op = lambda use_cuda: MagicMock(
            cpu_required=0,
            mem_required=0,
            gpu_required=None,
            num_proc=None,
            _name="test_op",
            use_cuda=lambda: use_cuda
        )
        
        # Common patchers
        self.ray_cpu_patcher = patch(
            'data_juicer.utils.ray_utils.ray_cpu_count')
        self.ray_gpu_patcher = patch(
            'data_juicer.utils.ray_utils.ray_gpu_count')
        self.ray_mem_patcher = patch(
            'data_juicer.utils.ray_utils.ray_available_memories')
        self.ray_gpu_mem_patcher = patch(
            'data_juicer.utils.ray_utils.ray_available_gpu_memories')
        self.cuda_available_patcher = patch(
            'data_juicer.utils.resource_utils.is_cuda_available')
        
        self.mock_cpu = self.ray_cpu_patcher.start()
        self.mock_gpu = self.ray_gpu_patcher.start()
        self.mock_mem = self.ray_mem_patcher.start()
        self.mock_gpu_mem = self.ray_gpu_mem_patcher.start()
        self.mock_cuda_available = self.cuda_available_patcher.start()

        # Default cluster resources (4 CPUs, 8GB RAM, 1 GPU 16GB)
        self.mock_cpu.return_value = 4
        self.mock_gpu.return_value = 1
        self.mock_mem.return_value = [8192]  # 8GB
        self.mock_gpu_mem.return_value = [16384]  # 16GB
        self.mock_cuda_available.return_value = True

    def tearDown(self):
        self.ray_cpu_patcher.stop()
        self.ray_gpu_patcher.stop()
        self.ray_mem_patcher.stop()
        self.ray_gpu_mem_patcher.stop()
        self.cuda_available_patcher.stop()

    def test_cpu_op_auto_scaling(self):
        """Test CPU operator with auto scaling"""
        op = self.mock_op(use_cuda=False)
        op.cpu_required = 1
        
        RayDataset.set_resource_for_ops([op])
        self.assertEqual(op.num_proc, 4)  # 4 CPUs / 1 per op
        self.assertEqual(op.cpu_required, 1)
        self.assertEqual(op.gpu_required, None)

    def test_gpu_op_auto_scaling(self):
        """Test GPU operator with auto scaling"""
        op = self.mock_op(use_cuda=True)
        op.gpu_required = 1
        
        RayDataset.set_resource_for_ops([op])
        self.assertEqual(op.num_proc, 1)  # Only 1 GPU available
        self.assertEqual(op.gpu_required, 1)
        self.assertEqual(op.cpu_required, 0)

    def test_user_specified_num_proc(self):
        """Test user-specified num_proc takes priority"""
        op = self.mock_op(use_cuda=False)
        op.num_proc = 2
        op.cpu_required = 1
        
        RayDataset.set_resource_for_ops([op])
        self.assertEqual(op.num_proc, 2)
        self.assertEqual(op.cpu_required, 1)
        self.assertEqual(op.gpu_required, None)

    def test_mixed_ops_resource_allocation(self):
        """Test mixed operators with fixed and auto scaling"""
        fixed_op = self.mock_op(use_cuda=False)
        fixed_op._name = 'op1'
        fixed_op.num_proc = (1, 2)
        fixed_op.cpu_required = 1
        
        auto_op = self.mock_op(use_cuda=False)
        auto_op._name = 'op2'
        auto_op.cpu_required = 1
        
        self.mock_cpu.return_value = 8  # 8 CPUs total
        
        RayDataset.set_resource_for_ops([fixed_op, auto_op])
        # Fixed min: 1 core * 1 proc = 1 core
        # Fixed max: 1 core * 2 proc = 2 core
        # Auto: each needs 1 core, remaining (8-2)-(8-1) cores
        # Min proc = 6 // 1 = 6, Max = 7 // 1 = 7
        self.assertEqual(auto_op.num_proc, 7)
        self.assertEqual(auto_op.cpu_required, 1)
        self.assertEqual(fixed_op.cpu_required, 1)
        self.assertEqual(fixed_op.num_proc, (1, 2))

    def test_insufficient_resources(self):
        """Test resource overallocation exception"""
        op1 = self.mock_op(use_cuda=False)
        op1._name = 'op1'
        op1.num_proc = 5  # 1 core per proc
        op1.cpu_required = 1
        
        op2 = self.mock_op(use_cuda=False)
        op2._name = 'op2'
        op2.cpu_required = 1
        
        self.mock_cpu.return_value = 4  # Only 4 cores available
        
        with self.assertRaises(ValueError) as cm:
            RayDataset.set_resource_for_ops([op1, op2])

        # required cpus: 1*5 + 1, 6/4=1.5
        self.assertEqual(str(cm.exception),
                         "Insufficient cluster resources: At least 1.50x the current resource is required. "
                         "Add resources or reduce operator requirements.")

    def test_gpu_op_without_cuda(self):
        """Test GPU operator when CUDA is unavailable"""
        self.mock_cuda_available.return_value = False
        op = self.mock_op(use_cuda=True)
        op.gpu_required = 1
        
        with self.assertRaises(ValueError) as cm:
            RayDataset.set_resource_for_ops([op])

        self.assertEqual(str(cm.exception), 
                         "Op[test_op] attempted to request GPU resources (gpu_required=1), "
                         "but the gpu is unavailable. Please check whether your environment is installed correctly"
                         " and whether there is a gpu in the resource pool.")

    def test_multi_ops_with_cpu_gpu(self):
        """Test operator with no resource requirements"""

        op1_cuda = self.mock_op(use_cuda=True)
        op1_cuda.mem_required = 2
        op1_cuda.cpu_required = 1
        op1_cuda._name = 'op1_cuda'

        op2_cuda = self.mock_op(use_cuda=True)
        op2_cuda.gpu_required = 0.5
        op2_cuda._name = 'op2_cuda'

        op3_cuda = self.mock_op(use_cuda=True)
        op3_cuda.gpu_required = 0.2
        op3_cuda.num_proc = (5, 10)
        op3_cuda._name = 'op3_cuda'

        op1_cpu = self.mock_op(use_cuda=False)
        op1_cpu.mem_required = 2
        op1_cpu._name = 'op1_cpu'

        op2_cpu = self.mock_op(use_cuda=False)
        op2_cpu.cpu_required = 0.5
        op2_cpu._name = 'op2_cpu'

        op3_cpu = self.mock_op(use_cuda=False)
        op3_cpu.cpu_required = 0.2
        op3_cpu.num_proc = 10
        op3_cpu._name = 'op3_cpu'

        op4_cpu = self.mock_op(use_cuda=False)
        op4_cpu._name = 'op4_cpu'

        self.mock_cpu.return_value = 100
        self.mock_gpu.return_value = 5
        self.mock_mem.return_value = [131072]  # 128 GB
        self.mock_gpu_mem.return_value = [51200]  # 5 * 10GB

        RayDataset.set_resource_for_ops([op1_cuda, op2_cuda, op3_cuda, op1_cpu, op2_cpu, op3_cpu, op4_cpu])

        self.assertEqual(op1_cuda.num_proc, (2, 13))
        self.assertEqual(op1_cuda.cpu_required, 1)
        self.assertEqual(op1_cuda.gpu_required, 0.29)
        self.assertEqual(op1_cuda.mem_required, 2)

        self.assertEqual(op2_cuda.num_proc, (2, 7))
        self.assertEqual(op2_cuda.cpu_required, 0)
        self.assertEqual(op2_cuda.gpu_required, 0.5)
        self.assertEqual(op2_cuda.mem_required, 0)

        self.assertEqual(op3_cuda.num_proc, (5, 10))
        self.assertEqual(op3_cuda.cpu_required, 0)
        self.assertEqual(op3_cuda.gpu_required, 0.2)
        self.assertEqual(op3_cuda.mem_required, 0)

        self.assertEqual(op1_cpu.num_proc, 34)
        self.assertEqual(op1_cpu.cpu_required, 0)
        self.assertEqual(op1_cpu.gpu_required, None)
        self.assertEqual(op1_cpu.mem_required, 2)

        self.assertEqual(op2_cpu.num_proc, 156)
        self.assertEqual(op2_cpu.cpu_required, 0.5)
        self.assertEqual(op2_cpu.gpu_required, None)
        self.assertEqual(op2_cpu.mem_required, 0)

        self.assertEqual(op3_cpu.num_proc, 10)
        self.assertEqual(op3_cpu.cpu_required, 0.2)
        self.assertEqual(op3_cpu.gpu_required, None)
        self.assertEqual(op3_cpu.mem_required, 0)

        self.assertEqual(op4_cpu.num_proc, 78)
        self.assertEqual(op4_cpu.cpu_required, 0)
        self.assertEqual(op4_cpu.gpu_required, None)
        self.assertEqual(op4_cpu.mem_required, 0)


if __name__ == '__main__':
    unittest.main()
