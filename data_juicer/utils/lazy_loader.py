"""A LazyLoader class for on-demand module loading with uv integration."""

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
import tomli
import time
import types
import re
import pkg_resources

from loguru import logger


class LazyLoader(types.ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies.
    Uses uv for fast dependency installation when needed.
    Automatically resolves dependencies from pyproject.toml.
    """

    # Special mappings for modules that have different package names
    MODULE_MAPPINGS = {
        'cv2': 'opencv-python',
        'aesthetics_predictor': 'simple-aesthetics-predictor',
    }

    def __init__(self, local_name, name, auto_install=True):
        """
        Initialize the LazyLoader.
        
        Args:
            local_name: The local name to use for the module
            name: The actual module name to import
            auto_install: Whether to automatically install missing dependencies
        """
        self._local_name = local_name
        self._auto_install = auto_install
        frame = inspect.currentframe().f_back
        self._parent_module_globals = frame.f_globals
        self._dependencies = self._load_dependencies()
        self._module = None
        logger.debug(f"Initialized LazyLoader for module: {name}")
        super(LazyLoader, self).__init__(name)

    def _handle_error(self, error, module_name):
        """Handle errors, including optional dependencies."""
        # Check if the error message indicates a missing optional dependency
        error_msg = str(error)
        
        # Try different patterns to extract the package name
        patterns = [
            r"requires `([^`]+)`",  # "requires `package`"
            r"install `([^`]+)`",   # "install `package`"
            r"install ([^\s]+)",    # "install package"
            r"requires ([^\s]+)",   # "requires package"
        ]
        
        # Then check for other dependencies
        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                dep_name = match.group(1)
                package_name = dep_name
                if package_name in self._dependencies:
                    logger.info(f"Installing optional dependency '{package_name}'...")
                    try:
                        subprocess.check_call([
                            sys.executable, '-m', 'uv', 'pip', 'install', self._dependencies[package_name]
                        ])
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        subprocess.check_call([
                            sys.executable, '-m', 'pip', 'install', self._dependencies[package_name]
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
            name = dep.split('[')[0].split('>')[0].split('<')[0].split('=')[0].strip()
            dependencies[name] = dep
            
        # Optional dependencies
        optional_deps = config.get('project', {}).get('optional-dependencies', {})
        for group, deps in optional_deps.items():
            for dep in deps:
                name = dep.split('[')[0].split('>')[0].split('<')[0].split('=')[0].strip()
                dependencies[name] = dep
                logger.debug(f"Found optional dependency: {name} in group {group}")
                
        return dependencies

    @staticmethod
    def check_packages(packages, extra_args=None):
        """
        Check if packages are installed and install them if needed.
        
        Args:
            packages (list): List of package names to check/install
            extra_args (str, optional): Additional arguments for installation
        """
        for package in packages:
            try:
                importlib.import_module(package.split('-')[0])
            except ImportError:
                logger.info(f"Package '{package}' not found. Installing...")
                try:
                    # Try uv first
                    cmd = [sys.executable, '-m', 'uv', 'pip', 'install', package]
                    if extra_args:
                        cmd.extend(extra_args.split())
                    subprocess.check_call(cmd)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback to pip if uv is not available
                    logger.warning("uv not found, falling back to pip...")
                    cmd = [sys.executable, '-m', 'pip', 'install', package]
                    if extra_args:
                        cmd.extend(extra_args.split())
                    subprocess.check_call(cmd)
                logger.info(f"Successfully installed {package}")

    def _get_package_name(self, module_name):
        """Get the package name for a module, handling special cases."""
        return self.MODULE_MAPPINGS.get(module_name, module_name)

    def _load(self):
        """Load the module and handle any missing dependencies."""
        if self._module is not None:
            return self._module

        try:
            # Try to import the module directly first
            self._module = importlib.import_module(self.__name__)
        except ImportError as e:
            if not self._auto_install:
                raise

            # Get the package name for installation
            package_name = self._get_package_name(self.__name__)
            
            # Try to install the package using uv first
            try:
                logger.info(f"Attempting to install {package_name} using uv...")
                subprocess.check_call([
                    sys.executable, '-m', 'uv', 'pip', 'install', package_name
                ])
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fall back to pip if uv fails
                logger.info(f"uv not available, falling back to pip for {package_name}...")
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', package_name
                    ])
                except subprocess.CalledProcessError as pip_error:
                    raise ImportError(
                        f"Failed to install {package_name}. This package may require system-level dependencies. "
                        f"Please try installing it manually with: pip install {package_name}\n"
                        f"Error details: {str(pip_error)}"
                    )

            # Try importing again - use the original module name, not the package name
            try:
                self._module = importlib.import_module(self._local_name)
            except ImportError as import_error:
                raise ImportError(
                    f"Failed to import {self._local_name} after installing {package_name}. "
                    f"Error details: {str(import_error)}"
                )

        # Update the parent module's globals with the loaded module
        self._parent_module_globals[self._local_name] = self._module
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

# For backward compatibility
class AUTOINSTALL:
    @staticmethod
    def check(packages, extra_args=None):
        """Alias for LazyLoader.check_packages for backward compatibility"""
        return LazyLoader.check_packages(packages, extra_args)
