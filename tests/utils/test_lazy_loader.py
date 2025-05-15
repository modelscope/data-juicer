import unittest
from types import ModuleType
from unittest.mock import patch
import subprocess
import os
import tempfile
from pathlib import Path
import shutil

from data_juicer.utils.lazy_loader import LazyLoader, get_toml_file_path
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LazyLoaderTest(DataJuicerTestCaseBase):
    """Test cases for basic LazyLoader functionality."""

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
        with patch('subprocess.check_call') as mock_check_call, \
             patch('importlib.import_module') as mock_import:
            # First import attempt fails
            mock_import.side_effect = ImportError("Module not found")
            # Installation attempt fails
            mock_check_call.side_effect = subprocess.CalledProcessError(1, 'cmd')
            
            loader = LazyLoader('test')
            with self.assertRaises(ImportError):
                loader._load()
            # Verify that installation was attempted
            mock_check_call.assert_called()


class LazyLoaderDependencyTest(DataJuicerTestCaseBase):
    """Test cases for LazyLoader dependency management."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Reset the dependencies cache before each test
        LazyLoader.reset_dependencies_cache()
        self.pyproject_content = """
[project]
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.26.4,<2.0.0",
    "loguru",
    "tqdm",
    "jsonargparse[signatures]",
    "jsonlines",
    "zstandard",
    "lz4",
    "multiprocess==0.70.12",
    "dill==0.3.4",
    "psutil",
    "pydantic>=2.0",
    "uv",
    "wordcloud",
    "spacy==3.7.0",
    "httpx",
    "av==13.1.0",
    "emoji==2.2.0",
    "tabulate",
    "librosa>=0.10",
    "resampy",
    "bs4",
    "matplotlib",
    "plotly",
    "seaborn",
    "requests",
    "wget",
    "pdfplumber",
    "python-docx",
    "streamlit",
    "Pillow",
    "fastapi>=0.110",
    "mwparserfromhell",
    "regex"
]

[project.optional-dependencies]
vision = [
    "opencv-python",
    "Pillow>=10.0.0",
    "torchvision"
]
nlp = [
    "transformers>=4.30.0",
    "sentencepiece",
    "kenlm",
    "fasttext-wheel"
]
audio = [
    "librosa>=0.10",
    "resampy",
    "soundfile"
]
"""
        self.pyproject_path = self.create_test_pyproject(self.pyproject_content)

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.pyproject_path):
            os.unlink(self.pyproject_path)
        super().tearDown()

    def create_test_pyproject(self, content):
        """Helper function to create a temporary pyproject.toml file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            return f.name

    def test_get_all_dependencies(self):
        """Test getting all dependencies from pyproject.toml."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path:
            mock_get_path.return_value = Path(self.pyproject_path)
            deps = LazyLoader.get_all_dependencies()
            
            # Check main dependencies
            self.assertEqual(deps['pandas'], 'pandas>=2.0.0')
            self.assertEqual(deps['numpy'], 'numpy>=1.26.4,<2.0.0')
            self.assertEqual(deps['loguru'], 'loguru')
            
            # Check optional dependencies
            self.assertEqual(deps['opencv-python'], 'opencv-python')
            self.assertEqual(deps['Pillow'], 'Pillow>=10.0.0')
            self.assertEqual(deps['transformers'], 'transformers>=4.30.0')
            self.assertEqual(deps['librosa'], 'librosa>=0.10')

    def test_get_all_dependencies_not_found(self):
        """Test behavior when pyproject.toml is not found."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path', side_effect=FileNotFoundError):
            deps = LazyLoader.get_all_dependencies()
            self.assertEqual(deps, {})

    def test_get_all_dependencies_parse_error(self):
        """Test behavior when pyproject.toml is invalid."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as f:
            f.write('invalid toml content')
            f.flush()
            
            with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path:
                mock_get_path.return_value = Path(f.name)
                deps = LazyLoader.get_all_dependencies()
                self.assertEqual(deps, {})

    def test_get_all_dependencies_empty_sections(self):
        """Test behavior when pyproject.toml has empty dependency sections."""
        empty_content = """
[project]
dependencies = []

[project.optional-dependencies]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as f:
            f.write(empty_content)
            f.flush()
            
            with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path:
                mock_get_path.return_value = Path(f.name)
                deps = LazyLoader.get_all_dependencies()
                self.assertEqual(deps, {})

    def test_get_all_dependencies_missing_sections(self):
        """Test behavior when pyproject.toml is missing dependency sections."""
        minimal_content = """
[project]
name = "data-juicer"
version = "0.1.0"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as f:
            f.write(minimal_content)
            f.flush()
            
            with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path:
                mock_get_path.return_value = Path(f.name)
                deps = LazyLoader.get_all_dependencies()
                self.assertEqual(deps, {})

    def test_install_package_with_version(self):
        """Test package installation with version from dependencies."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # Test installing a package that has a version in dependencies
            LazyLoader._install_package('pandas')
            
            # Verify that the correct version was used
            mock_check_call.assert_called()
            args, kwargs = mock_check_call.call_args
            self.assertIn('pandas>=2.0.0', args[0])

    def test_install_package_with_complex_version(self):
        """Test package installation with complex version constraints."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # Test installing a package with complex version constraints
            LazyLoader._install_package('numpy')
            
            # Verify that the correct version was used
            mock_check_call.assert_called()
            args, kwargs = mock_check_call.call_args
            self.assertIn('numpy>=1.26.4,<2.0.0', args[0])

    def test_install_package_with_extras(self):
        """Test package installation with extras."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # Test installing a package with extras
            LazyLoader._install_package('jsonargparse[signatures]')
            
            # Verify that the correct version was used
            mock_check_call.assert_called()
            args, kwargs = mock_check_call.call_args
            self.assertIn('jsonargparse[signatures]', args[0])

    def test_install_package_without_version(self):
        """Test package installation for packages without version in dependencies."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # Test installing a package that doesn't have a version in dependencies
            LazyLoader._install_package('nonexistent')
            
            # Verify that the original package name was used
            mock_check_call.assert_called()
            args, kwargs = mock_check_call.call_args
            self.assertIn('nonexistent', args[0])

    def test_install_package_with_github_url(self):
        """Test package installation with GitHub URL."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call, \
             patch('git.Repo.clone_from') as mock_clone:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # Test installing a package from GitHub
            LazyLoader._install_package('git+https://github.com/user/repo.git')
            
            # Verify that git clone was called
            mock_clone.assert_called_once()
            # Verify that the original URL was used
            args, kwargs = mock_clone.call_args
            self.assertEqual(args[0], 'https://github.com/user/repo.git')

    def test_install_package_with_cached_dependencies(self):
        """Test that package installation uses cached dependencies."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # First call should load dependencies
            LazyLoader._install_package('pandas')
            first_call_args = mock_check_call.call_args[0][0]
            
            # Reset mock to verify second call
            mock_check_call.reset_mock()
            
            # Second call should use cached dependencies
            LazyLoader._install_package('pandas')
            second_call_args = mock_check_call.call_args[0][0]
            
            # Verify both calls used the same version
            self.assertEqual(first_call_args, second_call_args)
            self.assertIn('pandas>=2.0.0', first_call_args)
            
            # Verify get_toml_file_path was only called once
            self.assertEqual(mock_get_path.call_count, 1)

    def test_install_package_with_optional_dependency(self):
        """Test installing a package from optional dependencies."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # Test installing a package from optional dependencies
            LazyLoader._install_package('opencv-python')
            
            # Verify that the correct version was used
            mock_check_call.assert_called()
            args, kwargs = mock_check_call.call_args
            self.assertIn('opencv-python', args[0])

    def test_install_package_with_version_override(self):
        """Test that package URL overrides version from dependencies."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call, \
             patch('git.Repo.clone_from') as mock_clone, \
             patch('os.path.exists') as mock_exists, \
             patch('shutil.rmtree') as mock_rmtree:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            # Mock requirements.txt not existing
            mock_exists.return_value = False
            
            # Test installing a package with a URL that should override the version
            LazyLoader._install_package('git+https://github.com/user/pandas.git')
            
            # Verify that git clone was called with the correct URL
            mock_clone.assert_called_once()
            args, kwargs = mock_clone.call_args
            self.assertEqual(args[0], 'https://github.com/user/pandas.git')
            
            # Verify that the package was installed in editable mode
            mock_check_call.assert_called()
            args, kwargs = mock_check_call.call_args
            self.assertIn('-e', args[0])
            
            # Verify cleanup was called
            mock_rmtree.assert_called_once()

    def test_install_package_with_version_constraint(self):
        """Test package installation when version constraint is found in dependencies."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call, \
             patch('loguru.logger.info') as mock_info:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # Test installing a package that has a version in dependencies
            LazyLoader._install_package('pandas')
            
            # Verify that the correct version was used
            mock_check_call.assert_called()
            args, kwargs = mock_check_call.call_args
            self.assertIn('pandas>=2.0.0', args[0])
            
            # Verify that an info message was logged
            mock_info.assert_called()
            info_msg = mock_info.call_args[0][0]
            self.assertIn('Installing pandas>=2.0.0 using uv', info_msg)

    def test_install_package_without_version_constraint(self):
        """Test package installation when no version constraint is found in dependencies."""
        with patch('data_juicer.utils.lazy_loader.get_toml_file_path') as mock_get_path, \
             patch('subprocess.check_call') as mock_check_call, \
             patch('loguru.logger.warning') as mock_warning:
            # Set up the mock pyproject.toml
            mock_get_path.return_value = Path(self.pyproject_path)
            
            # Test installing a package that doesn't exist in dependencies
            LazyLoader._install_package('nonexistent-package')
            
            # Verify that the original package name was used
            mock_check_call.assert_called()
            args, kwargs = mock_check_call.call_args
            self.assertIn('nonexistent-package', args[0])
            
            # Verify that a warning was logged
            mock_warning.assert_called()
            warning_msg = mock_warning.call_args[0][0]
            self.assertIn('No version constraint found in pyproject.toml', warning_msg)
            self.assertIn('nonexistent-package', warning_msg)


if __name__ == '__main__':
    unittest.main()
