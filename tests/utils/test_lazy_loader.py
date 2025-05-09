import unittest
from types import ModuleType
from unittest.mock import patch
import inspect
import subprocess

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LazyLoaderTest(DataJuicerTestCaseBase):

    def test_basic_func(self):
        torch = LazyLoader('torch')
        # it's a LazyLoader at the beginning
        self.assertIsInstance(torch, LazyLoader)
        # invoke it or check the dir to install and activate it
        self.assertIsInstance(dir(torch), list)
        # it's a real module now
        self.assertIsInstance(torch, ModuleType)

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
        numpy1 = LazyLoader('numpy')
        numpy2 = LazyLoader('numpy')
        # Both should be LazyLoader instances initially
        self.assertIsInstance(numpy1, LazyLoader)
        self.assertIsInstance(numpy2, LazyLoader)
        # Access numpy1 to load it
        dir(numpy1)
        # numpy2 should still be a LazyLoader
        self.assertIsInstance(numpy2, LazyLoader)

    def test_package_url_format(self):
        """Test package URL format handling."""
        # Test with separate package_url
        loader2 = LazyLoader('test', 'package', package_url='git+https://github.com/user/repo.git')
        self.assertEqual(loader2._package_name, 'package')
        self.assertEqual(loader2._package_url, 'git+https://github.com/user/repo.git')

        # Test with no URL
        loader3 = LazyLoader('test', 'package')
        self.assertEqual(loader3._package_name, 'package')
        self.assertIsNone(loader3._package_url)

    def test_github_installation(self):
        """Test GitHub package installation."""
        with patch('subprocess.check_call') as mock_check_call:
            try:
                loader = LazyLoader('test', package_url='git+https://github.com/user/repo.git')
                dir(loader)  # This should trigger installation
            except ImportError:
                # This is expected since the repo doesn't exist
                # Verify git clone was called with the correct URL
                mock_check_call.assert_called_once()
                args, kwargs = mock_check_call.call_args
                self.assertTrue('git' in args[0])
                self.assertTrue('clone' in args[0])
                self.assertTrue('https://github.com/user/repo.git' in args[0])
                pass

    def test_dependency_handling(self):
        """Test handling of optional dependencies."""
        with patch('subprocess.check_call') as mock_check_call:
            try:
                loader = LazyLoader('test')
                # Simulate a missing dependency error
                with patch.object(loader, '_handle_error', return_value=True):
                    loader._load()
            except ImportError:
                # This is expected since the package doesn't exist
                mock_check_call.assert_called()
                pass


if __name__ == '__main__':
    unittest.main()
