import unittest
from types import ModuleType
from unittest.mock import patch
import inspect
import subprocess

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

    def test_module_mappings(self):
        # Test special module mappings
        cv2 = LazyLoader('cv2', 'cv2')
        self.assertEqual(cv2._get_package_name('cv2'), 'opencv-python')
        
        aesthetics = LazyLoader('aesthetics', 'aesthetics_predictor')
        self.assertEqual(aesthetics._get_package_name('aesthetics_predictor'), 'simple-aesthetics-predictor')

    def test_auto_install_disabled(self):
        # Test with auto_install=False
        with self.assertRaises(ImportError):
            nonexistent = LazyLoader('nonexistent', 'nonexistent_module', auto_install=False)
            dir(nonexistent)

    def test_dependency_loading(self):
        # Test dependency loading from pyproject.toml
        loader = LazyLoader('test', 'test')
        deps = loader._load_dependencies()
        self.assertIsInstance(deps, dict)
        # Check if it contains some expected dependencies
        self.assertTrue(any('torch' in key for key in deps.keys()))

    def test_error_handling(self):
        # Test error handling for missing dependencies
        with patch('subprocess.check_call') as mock_check_call:
            mock_check_call.side_effect = subprocess.CalledProcessError(1, 'cmd')
            with self.assertRaises(ImportError):
                nonexistent = LazyLoader('nonexistent', 'nonexistent_module')
                dir(nonexistent)

    def test_check_packages(self):
        # Test the static check_packages method
        # Note: The package name should be a valid PyPI package name
        # that matches its module name (e.g., 'numpy' instead of 'test-package')
        with patch('subprocess.check_call') as mock_check_call:
            LazyLoader.check_packages(['testpackage'])
            mock_check_call.assert_called()

    def test_uv_fallback(self):
        # Test fallback to pip when uv is not available
        with patch('subprocess.check_call') as mock_check_call:
            # Make FileNotFoundError persist for all calls
            mock_check_call.side_effect = FileNotFoundError
            # Should fall back to pip
            with self.assertRaises(FileNotFoundError):
                LazyLoader.check_packages(['testpackage'])
            # Verify that both uv and pip were attempted
            self.assertEqual(mock_check_call.call_count, 2)
            # First call should be uv
            self.assertIn('uv', mock_check_call.call_args_list[0][0][0])
            # Second call should be pip
            self.assertIn('pip', mock_check_call.call_args_list[1][0][0])

    def test_module_attributes(self):
        # Test accessing module attributes
        numpy = LazyLoader('numpy', 'numpy')
        # Access an attribute to trigger loading
        self.assertIsNotNone(numpy.array)

    def test_module_dir(self):
        # Test dir() functionality
        numpy = LazyLoader('numpy', 'numpy')
        dir_list = dir(numpy)
        self.assertIsInstance(dir_list, list)
        self.assertTrue('array' in dir_list)

    def test_multiple_instances(self):
        # Test multiple instances of the same module
        numpy1 = LazyLoader('numpy1', 'numpy')
        numpy2 = LazyLoader('numpy2', 'numpy')
        # Both should be LazyLoader instances initially
        self.assertIsInstance(numpy1, LazyLoader)
        self.assertIsInstance(numpy2, LazyLoader)
        # Access numpy1 to load it
        dir(numpy1)
        # numpy2 should still be a LazyLoader
        self.assertIsInstance(numpy2, LazyLoader)

    def test_parent_module_globals(self):
        # Test that the module is added to parent globals
        # Create a LazyLoader in the current scope
        numpy = LazyLoader('numpy3', 'numpy')
        # Module should not be in the current scope's globals before trigger
        self.assertNotIn('numpy3', globals())
        # Access the module to trigger loading
        dir(numpy)
        # Module should be in the current scope's globals now
        self.assertIn('numpy3', globals())
        # The module should be the same instance
        self.assertIs(globals()['numpy3'], numpy._module)

if __name__ == '__main__':
    unittest.main()
