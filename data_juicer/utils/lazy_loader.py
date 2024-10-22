"""A LazyLoader class."""

import importlib
import inspect
import os
import types

from loguru import logger

from data_juicer.utils.auto_install_utils import (AutoInstaller,
                                                  _is_module_installed)
from data_juicer.utils.availability_utils import _torch_check_and_set

current_path = os.path.dirname(os.path.realpath(__file__))
science_file_path = os.path.join(current_path,
                                 '../../environments/science_requires.txt')
dist_file_path = os.path.join(current_path,
                              '../../environments/dist_requires.txt')
AUTOINSTALL = AutoInstaller([science_file_path, dist_file_path])


class LazyLoader(types.ModuleType):
    """
    Lazily import a module, mainly to avoid pulling in large dependencies.
    `contrib`, and `ffmpeg` are examples of modules that are large and not
    always needed, and this allows them to only be loaded when they are used.
    """

    # The lint error here is incorrect.
    def __init__(self, local_name, name, auto_install=True):
        self._local_name = local_name
        # get last frame in the stack
        frame = inspect.currentframe().f_back
        # get the globals of module who calls LazyLoader
        self._parent_module_globals = frame.f_globals
        self.auto_install = auto_install

        super(LazyLoader, self).__init__(name)

    def _load(self):
        # Auto install if necessary
        module_name = self.__name__.split('.')[0]
        if self.auto_install and not _is_module_installed(module_name):
            logger.warning(
                f"Module '{module_name}' not installed or fully installed.")
            logger.warning(f"Auto installing '{module_name}' ...")
            AUTOINSTALL.install(module_name)
        # check for torch
        if self.__name__ == 'torch':
            _torch_check_and_set()
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)

        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on
        #   lookups that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
