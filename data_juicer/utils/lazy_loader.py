"""A LazyLoader class for on-demand module loading with uv integration."""

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
import tomli
import time
import types

from loguru import logger

class LazyLoader(types.ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies.
    Uses uv for fast dependency installation when needed.
    Automatically resolves dependencies from pyproject.toml.
    """

    def __init__(self, local_name, name):
        self._local_name = local_name
        frame = inspect.currentframe().f_back
        self._parent_module_globals = frame.f_globals
        self._dependencies = self._load_dependencies()
        logger.debug(f"Initialized LazyLoader for module: {name}")
        super(LazyLoader, self).__init__(name)

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

    def _load(self):
        # Get the base module name
        module_name = self.__name__.split('.')[0]
        logger.debug(f"Attempting to load module: {module_name}")
        
        try:
            # Try to import first
            start_time = time.time()
            module = importlib.import_module(self.__name__)
            load_time = time.time() - start_time
            logger.debug(f"Successfully imported {module_name} in {load_time:.2f}s")
        except ImportError:
            # If import fails, check if we have the dependency
            if module_name in self._dependencies:
                package = self._dependencies[module_name]
                logger.info(f"Module '{module_name}' not installed. Installing package '{package}'...")
                start_time = time.time()
                try:
                    # Try uv first
                    logger.debug("Attempting installation with uv...")
                    subprocess.check_call([
                        sys.executable, '-m', 'uv', 'pip', 'install', package
                    ])
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback to pip if uv is not available
                    logger.warning("uv not found, falling back to pip...")
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', package
                    ])
                install_time = time.time() - start_time
                logger.info(f"Package '{package}' installed in {install_time:.2f}s")
                
                # Now try importing again
                start_time = time.time()
                module = importlib.import_module(self.__name__)
                import_time = time.time() - start_time
                logger.info(f"Successfully imported {module_name} in {import_time:.2f}s")
            else:
                # If we don't have the dependency, just try to import
                logger.debug(f"Module '{module_name}' not found in dependencies, attempting direct import")
                module = importlib.import_module(self.__name__)

        self._parent_module_globals[self._local_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        logger.debug(f"Accessing attribute '{item}' from lazy-loaded module {self.__name__}")
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
