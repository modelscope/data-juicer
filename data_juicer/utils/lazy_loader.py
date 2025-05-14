"""A LazyLoader class for on-demand module loading with uv integration."""

import importlib
import inspect
import subprocess
import sys
import traceback
import types

from loguru import logger


class LazyLoader(types.ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies.
    Uses uv for fast dependency installation when available.
    """

    @classmethod
    def check_packages(cls, package_specs, pip_args=None):
        """
        Check if packages are installed and install them if needed.

        Args:
            package_specs: A list of package specifications to check/install.
                          Can be package names or URLs (e.g., 'torch' or 'git+https://github.com/...')
            pip_args: Optional list of additional arguments to pass to pip install command
                     (e.g., ['--no-deps', '--upgrade'])
        """

        def _is_package_installed(package_name):
            """Check if a package is installed by attempting to import it."""
            if '@' in package_name:
                package_name = package_name.split('@')[0]
            if '[' in package_name:
                package_name = package_name.split('[')[0]
            if '/' in package_name:  # Handle GitHub URLs
                package_name = package_name.split('/')[-1].replace('.git', '')
            try:
                importlib.import_module(package_name)
                return True
            except ImportError:
                return False

        # Convert pip_args to list if it's a string
        if isinstance(pip_args, str):
            pip_args = [pip_args]

        for package_spec in package_specs:
            if not _is_package_installed(package_spec):
                logger.info(f'Package {package_spec} not found, installing...')
                try:
                    cls._install_package(package_spec, pip_args)
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f'Failed to install {package_spec}. This package may '
                        f'require system-level dependencies. Please try '
                        f'installing it manually with: pip install {package_spec}\n'
                        f'Error details: {str(e)}')

    def __init__(self,
                 module_name: str,
                 package_name: str = None,
                 package_url: str = None,
                 auto_install: bool = True):
        """
        Initialize the LazyLoader.

        Args:
            module_name: The name of the module to import (e.g., 'cv2', 'ray.data', 'torchvision.models')
            package_name: The name of the pip package to install (e.g., 'opencv-python', 'ray', 'torchvision')
                        If None, will use the base module name (e.g., 'ray' for 'ray.data')
            package_url: The URL to install the package from (e.g., git+https://github.com/...)
            auto_install: Whether to automatically install missing dependencies
        """
        self._module_name = module_name

        # For installation, use the provided package_name or base module name
        if package_name is None:
            self._package_name = module_name.split('.')[0]
        else:
            self._package_name = package_name

        # Standardize package_url to use git+ format
        if package_url and '@' in package_url:
            # Convert from package@git+ format to git+ format
            self._package_url = package_url.split('@', 1)[1]
        else:
            self._package_url = package_url

        self._auto_install = auto_install

        frame = inspect.currentframe().f_back
        self._parent_module_globals = frame.f_globals
        self._module = None

        # Print trace information
        logger.debug(
            f'Initialized LazyLoader for module: {module_name} '
            f'(package: {self._package_name}' +
            (f', url: {self._package_url}' if self._package_url else '') + ')')
        # Get last 3 frames of the stack trace
        stack = traceback.extract_stack(frame)[-3:]
        logger.debug('LazyLoader called from:\n' +
                     ''.join(traceback.format_list(stack)))

        super(LazyLoader, self).__init__(module_name)

    @classmethod
    def _install_package(cls, package_spec, pip_args=None):
        """Install a package using uv if available, otherwise pip."""
        # Print trace information for package installation
        logger.debug(f'Installing package: {package_spec}')
        # Get last 3 frames of the stack trace
        stack = traceback.extract_stack()[-3:]
        logger.debug('Package installation triggered from:\n' +
                     ''.join(traceback.format_list(stack)))

        # Convert pip_args to list if it's a string
        if isinstance(pip_args, str):
            pip_args = [pip_args]

        # For GitHub repositories, clone first then install locally
        if package_spec.startswith(('git+', 'https://github.com/')):
            import os
            import shutil
            import tempfile

            import git

            # Create a temporary directory for cloning
            temp_dir = tempfile.mkdtemp()
            try:
                # Clone the repository
                logger.info(f'Cloning {package_spec}...')
                if package_spec.startswith('git+'):
                    repo_url = package_spec[4:]  # Remove 'git+' prefix
                else:
                    repo_url = package_spec
                git.Repo.clone_from(repo_url, temp_dir)

                # Check for requirements.txt and install dependencies first
                requirements_path = os.path.join(temp_dir, 'requirements.txt')
                if os.path.exists(requirements_path):
                    logger.info(
                        'Installing requirements from requirements.txt...')
                    try:
                        # Try uv first
                        cmd = [
                            sys.executable, '-m', 'uv', 'pip', 'install', '-r',
                            requirements_path
                        ]
                        if pip_args:
                            cmd.extend(pip_args)
                        subprocess.check_call(cmd)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        logger.warning(
                            'uv not found or failed, falling back to pip...')
                        cmd = [
                            sys.executable, '-m', 'pip', 'install', '-r',
                            requirements_path
                        ]
                        if pip_args:
                            cmd.extend(pip_args)
                        subprocess.check_call(cmd)

                # Install the package in editable mode
                try:
                    logger.info('Installing package in editable mode...')
                    cmd = [
                        sys.executable, '-m', 'uv', 'pip', 'install', '-e',
                        temp_dir
                    ]
                    if pip_args:
                        cmd.extend(pip_args)
                    subprocess.check_call(cmd)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning(
                        'uv not found or failed, falling back to pip...')
                    cmd = [
                        sys.executable, '-m', 'pip', 'install', '-e', temp_dir
                    ]
                    if pip_args:
                        cmd.extend(pip_args)
                    subprocess.check_call(cmd)
                return True
            finally:
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
        else:
            # For non-GitHub packages, use direct installation
            try:
                logger.info(f'Installing {package_spec} using uv...')
                cmd = [
                    sys.executable, '-m', 'uv', 'pip', 'install', package_spec
                ]
                if pip_args:
                    cmd.extend(pip_args)
                subprocess.check_call(cmd)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning(
                    'uv not found or failed, falling back to pip...')
                cmd = [sys.executable, '-m', 'pip', 'install', package_spec]
                if pip_args:
                    cmd.extend(pip_args)
                subprocess.check_call(cmd)
                return True

    def _load(self):
        """Load the module and handle any missing dependencies."""
        logger.debug(f'Loading {self._module_name}...')

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

            # Install the package
            try:
                self._install_package(package_spec)
            except subprocess.CalledProcessError as e:
                raise ImportError(
                    f'Failed to install {package_spec}. This package may '
                    f'require system-level dependencies. Please try '
                    f'installing it manually with: pip install {package_spec}\n'
                    f'Error details: {str(e)}')

            # Try importing again
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
