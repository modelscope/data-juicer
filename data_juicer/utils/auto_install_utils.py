import os
import sys
import subprocess
import importlib
import platform

from loguru import logger

from data_juicer.utils.availability_utils import _torch_check_and_set

def _is_package_installed(package_name):
    try:
        subprocess.check_output([sys.executable, '-m', 'pip', 'show', package_name], stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False

class AutoInstaller(object):
    """
    This class is used to install the required
    package automatically.
    """
    def __init__(self, require_f_paths=[]):
        """
        Initialization method.

        :param require_f_paths: paths to the file for version limitation
        """
        self.version_map, reqs = {}, []
        for path in require_f_paths:
            if not os.path.exists(path):
                logging.warning(f'target file does not exist: {path}')
            else:
                with open(path, 'r', encoding='utf-8') as fin:
                    reqs += [x.strip() for x in fin.read().splitlines()]
        for req in reqs:
            clean_req = req.replace('<', ' ').replace('>', ' ').replace('=', ' ').split(' ')[0]
            self.version_map[clean_req] = req

    def check(self, check_pkgs):
        """
        install if the package is not importable.
        """
        def wrapper(func):
            def inner_wrapper(*args, **kwargs):
                for pkg in check_pkgs:
                    if not _is_package_installed(pkg):
                        logger.info(f"Installing {pkg} ...")
                        if pkg in self.version_map:
                            pkg = self.version_map[pkg]
                        subprocess.check_call(["pip", "install", pkg])
                        logger.info(f"The {pkg} installed.")
                        if pkg == torch:
                            _torch_check_and_set()
                return func(*args, **kwargs)
            return inner_wrapper
        return wrapper
