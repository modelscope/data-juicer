"""A LazyLoader class for on-demand module loading with uv integration."""

import importlib
import inspect
import re
import subprocess
import sys
import types
from pathlib import Path

import tomli
from loguru import logger


class LazyLoader(types.ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies.
    Uses uv for fast dependency installation when needed.
    """

    def __init__(self,
                 module_name: str,
                 package_name: str = None,
                 auto_install: bool = True):
        """
        Initialize the LazyLoader.

        Args:
            module_name: The name of the module to import (e.g., 'cv2', 'ffmpeg')
            package_name: The name of the pip package to install (e.g., 'opencv-python', 'ffmpeg-python')
                        If None, will use module_name as package_name
            auto_install: Whether to automatically install missing dependencies
        """
        self._module_name = module_name
        self._package_name = package_name or module_name
        self._auto_install = auto_install
        frame = inspect.currentframe().f_back
        self._parent_module_globals = frame.f_globals
        self._dependencies = self._load_dependencies()
        self._module = None
        logger.debug(
            f'Initialized LazyLoader for module: {module_name} (package: {self._package_name})'
        )
        super(LazyLoader, self).__init__(module_name)

    def _handle_error(self, error, module_name):
        """Handle errors, including optional dependencies."""
        # Check if the error message indicates a missing optional dependency
        error_msg = str(error)

        # Try different patterns to extract the package name
        patterns = [
            r'requires `([^`]+)`',  # "requires `package`"
            r'install `([^`]+)`',  # "install `package`"
            r'install ([^\s]+)',  # "install package"
            r'requires ([^\s]+)',  # "requires package"
        ]

        # Then check for other dependencies
        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                dep_name = match.group(1)
                package_name = dep_name
                if package_name in self._dependencies:
                    logger.info(f'Installing optional dependency '
                                f"{package_name}'...")
                    try:
                        subprocess.check_call([
                            sys.executable, '-m', 'uv', 'pip', 'install',
                            self._dependencies[package_name]
                        ])
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        subprocess.check_call([
                            sys.executable, '-m', 'pip', 'install',
                            self._dependencies[package_name]
                        ])
                    return True

        return False

    def _load_dependencies(self):
        """Load dependencies from pyproject.toml"""
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / 'pyproject.toml'

        if not pyproject_path.exists():
            logger.warning(f'pyproject.toml not found at {pyproject_path}')
            return {}

        with open(pyproject_path, 'rb') as f:
            config = tomli.load(f)

        # Get all dependencies
        dependencies = {}

        # Main dependencies
        main_deps = config.get('project', {}).get('dependencies', [])
        for dep in main_deps:
            name = dep.split('[')[0].split('>')[0].split('<')[0].split(
                '=')[0].strip()  # noqa: E501
            dependencies[name] = dep

        # Optional dependencies
        optional_deps = config.get('project', {}).get('optional-dependencies',
                                                      {})  # noqa: E501
        for group, deps in optional_deps.items():
            for dep in deps:
                name = dep.split('[')[0].split('>')[0].split('<')[0].split(
                    '=')[0].strip()  # noqa: E501
                dependencies[name] = dep
                logger.debug(f'Found optional dependency: {name} '
                             f'in group {group}')

        return dependencies

    @staticmethod
    def check_packages(packages, extra_args=None):
        """
        Check if modules are installed and install them if needed.

        Args:
            packages (list): List of module names to check/install
            extra_args (str, optional): Additional arguments for installation

        Note:
            For modules where the package name differs from the module name,
            use the MODULE_MAPPINGS dictionary to specify the correct mapping.
            For example: 'cv2' -> 'opencv-python'
        """
        for module_name in packages:
            # First try to import the module directly
            try:
                importlib.import_module(module_name)
                continue
            except ImportError:
                pass

            # Install the package
            logger.info(
                f"Module '{module_name}' not found. Installing package "
                f"{module_name}'...")
            try:
                # Try uv first
                cmd = [
                    sys.executable, '-m', 'uv', 'pip', 'install', module_name
                ]
                if extra_args:
                    cmd.extend(extra_args.split())
                subprocess.check_call(cmd)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to pip if uv is not available
                logger.warning('uv not found, falling back to pip...')
                cmd = [sys.executable, '-m', 'pip', 'install', module_name]
                if extra_args:
                    cmd.extend(extra_args.split())
                subprocess.check_call(cmd)
            logger.info(f'Successfully installed {module_name}')

            # After installation, try importing the module again
            try:
                importlib.import_module(module_name)
            except ImportError:
                logger.warning(
                    f'Package {module_name} was installed but module '
                    f'{module_name} could not be imported. This might '
                    f'indicate a mismatch between the package name and '
                    f'module name.')

    def _load(self):
        """Load the module and handle any missing dependencies."""
        if self._module is not None:
            return self._module

        try:
            # Try to import the module directly first
            self._module = importlib.import_module(self._module_name)
        except ImportError:
            if not self._auto_install:
                raise

            # Try to install the package using uv first
            try:
                logger.info(
                    f'Attempting to install {self._package_name} using uv...')
                # Try using uv directly first
                try:
                    subprocess.check_call(
                        ['uv', 'pip', 'install', self._package_name])
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # If uv command fails, try using python -m uv
                    subprocess.check_call([
                        sys.executable, '-m', 'uv', 'pip', 'install',
                        self._package_name
                    ])
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fall back to pip if uv fails
                logger.info(
                    f'uv not available, falling back to pip for {self._package_name}...'
                )
                try:
                    # Try using pip directly first
                    try:
                        subprocess.check_call(
                            ['pip', 'install', self._package_name])
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # If pip command fails, try using python -m pip
                        subprocess.check_call([
                            sys.executable, '-m', 'pip', 'install',
                            self._package_name
                        ])
                except subprocess.CalledProcessError as pip_error:
                    raise ImportError(
                        f'Failed to install {self._package_name}. This package may '
                        f'require system-level dependencies. Please try '
                        f'installing it manually with: pip install {self._package_name}\n'
                        f'Error details: {str(pip_error)}')

            # Try importing again - use the original module name
            try:
                self._module = importlib.import_module(self._module_name)
            except ImportError as import_error:
                raise ImportError(
                    f'Failed to import {self._module_name} after '
                    f'installing {self._package_name}. '
                    f'Error details: {str(import_error)}')

        # Update the parent module's globals with the loaded module
        self._parent_module_globals[self._module_name] = self._module
        self.__dict__.update(self._module.__dict__)
        return self._module

    def __getattr__(self, item):
        if self._module is None:
            self._load()
        return getattr(self._module, item)

    def __dir__(self):
        if self._module is None:
            self._load()
        return dir(self._module)
