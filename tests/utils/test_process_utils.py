import unittest
import torch
import multiprocess as mp

from data_juicer.utils.process_utils import setup_mp, get_min_cuda_memory, calculate_np
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

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


if __name__ == '__main__':
    unittest.main()
