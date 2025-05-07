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
                 package_url: str = None,
                 auto_install: bool = True):
        """
        Initialize the LazyLoader.

        Args:
            module_name: The name of the module to import (e.g., 'cv2', 'ffmpeg')
            package_name: The name of the pip package to install (e.g., 'opencv-python', 'ffmpeg-python')
                        If None, will use module_name as package_name.
                        Can also be in format 'package@url' for URL-based installations.
            package_url: The URL to install the package from (e.g., git+https://github.com/...)
                        If package_name contains '@', this parameter is ignored.
            auto_install: Whether to automatically install missing dependencies
        """
        self._module_name = module_name

        # Handle package_name in format 'package@url'
        if package_name and '@' in package_name:
            self._package_name, self._package_url = package_name.split('@', 1)
        else:
            self._package_name = package_name or module_name
            self._package_url = package_url

        self._auto_install = auto_install
        frame = inspect.currentframe().f_back
        self._parent_module_globals = frame.f_globals
        self._dependencies = self._load_dependencies()
        self._module = None
        logger.debug(
            f'Initialized LazyLoader for module: {module_name} '
            f'(package: {self._package_name}' +
            (f', url: {self._package_url}' if self._package_url else '') + ')')
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

    def _install_github_deps(self, package_spec, use_uv=True):
        """Install dependencies for a GitHub package."""
        repo_path = package_spec.split('github.com/')[1].split('.git')[0]

        # Clone the repo to a temp directory
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone the repo
                subprocess.check_call([
                    'git', 'clone', f'https://github.com/{repo_path}.git',
                    temp_dir
                ])

                # Install the package with dependencies
                if use_uv:
                    subprocess.check_call(['uv', 'pip', 'install', '.'],
                                          cwd=temp_dir)
                else:
                    subprocess.check_call(['pip', 'install', '.'],
                                          cwd=temp_dir)
                return True
            except Exception as e:
                logger.warning(
                    f'Failed to install dependencies for {package_spec}: {str(e)}'
                )
                return False

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

            # Prepare the package spec for installation
            package_spec = self._package_url if self._package_url else self._package_name

            # Try to install the package using uv first
            try:
                logger.info(
                    f'Attempting to install {package_spec} using uv...')
                # Try using uv directly first
                try:
                    # For GitHub packages, install with dependencies
                    if 'git+' in package_spec:
                        # Install package with dependencies
                        self._install_github_deps(package_spec, use_uv=True)
                    else:
                        subprocess.check_call(
                            ['uv', 'pip', 'install', package_spec])
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # If uv command fails, try using python -m uv
                    if 'git+' in package_spec:
                        # Install package with dependencies
                        self._install_github_deps(package_spec, use_uv=True)
                    else:
                        subprocess.check_call([
                            sys.executable, '-m', 'uv', 'pip', 'install',
                            package_spec
                        ])
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fall back to pip if uv fails
                logger.info(
                    f'uv not available, falling back to pip for {package_spec}...'
                )
                try:
                    # Try using pip directly first
                    try:
                        if 'git+' in package_spec:
                            # Install package with dependencies
                            self._install_github_deps(package_spec,
                                                      use_uv=False)
                        else:
                            subprocess.check_call(
                                ['pip', 'install', package_spec])
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # If pip command fails, try using python -m pip
                        if 'git+' in package_spec:
                            # Install package with dependencies
                            self._install_github_deps(package_spec,
                                                      use_uv=False)
                        else:
                            subprocess.check_call([
                                sys.executable, '-m', 'pip', 'install',
                                package_spec
                            ])
                except subprocess.CalledProcessError as pip_error:
                    raise ImportError(
                        f'Failed to install {package_spec}. This package may '
                        f'require system-level dependencies. Please try '
                        f'installing it manually with: pip install {package_spec}\n'
                        f'Error details: {str(pip_error)}')

            # Try importing again - use the module name
            try:
                self._module = importlib.import_module(self._module_name)
            except ImportError as import_error:
                raise ImportError(
                    f'Failed to import {self._module_name} after '
                    f'installing {package_spec}. '
                    f'Error details: {str(import_error)}')

        # Update the parent module's globals with the loaded module
        self._parent_module_globals[self._module_name] = self._module
        self.__dict__.update(self._module.__dict__)
        return self._module

    def __getattr__(self, item):
        """Handle attribute access, including submodule imports."""
        if self._module is None:
            self._load()

        # Try to get the attribute directly
        try:
            return getattr(self._module, item)
        except AttributeError:
            # If not found, try importing it as a submodule
            try:
                submodule = importlib.import_module(
                    f'{self._module_name}.{item}')
                setattr(self._module, item, submodule)
                return submodule
            except ImportError:
                raise AttributeError(
                    f"module '{self._module_name}' has no attribute '{item}'")

    def __dir__(self):
        if self._module is None:
            self._load()
        return dir(self._module)
