import unittest

from types import ModuleType

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class LazyLoaderTest(DataJuicerTestCaseBase):

    def test_basic_func(self):
        torch = LazyLoader('torch', 'torch')
        # it's a LazyLoader at the beginning
        self.assertIsInstance(torch, LazyLoader)
        # invoke it or check the dir to install and activate it
        self.assertIsInstance(dir(torch), list)
        # it's a real module now
        self.assertIsInstance(torch, ModuleType)


if __name__ == '__main__':
    unittest.main()
