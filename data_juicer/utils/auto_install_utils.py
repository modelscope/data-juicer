import importlib
import os
import platform
import subprocess
import sys

from loguru import logger

CHECK_SYSTEM_INFO_ONCE = True


def _is_package_installed(package_name):
    if '@' in package_name:
        package_name = package_name.split('@')[0]
    if '[' in package_name:
        package_name = package_name.split('[')[0]
    try:
        subprocess.check_output(
            [sys.executable, '-m', 'pip', 'show', package_name],
            stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False


def _torch_check_and_set():
    # only for python3.8 on mac
    global CHECK_SYSTEM_INFO_ONCE
    if CHECK_SYSTEM_INFO_ONCE and importlib.util.find_spec(
            'torch') is not None:
        major, minor = sys.version_info[:2]
        system = platform.system()
        if major == 3 and minor == 8 and system == 'Darwin':
            logger.warning(
                'The torch.set_num_threads function does not '
                'work in python3.8 version on Mac systems. We will set '
                'OMP_NUM_THREADS to 1 manually before importing torch')

            os.environ['OMP_NUM_THREADS'] = str(1)
            CHECK_SYSTEM_INFO_ONCE = False
        import torch

        # avoid hanging when calling clip in multiprocessing
        torch.set_num_threads(1)


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
                logger.warning(f'target file does not exist: {path}')
            else:
                with open(path, 'r', encoding='utf-8') as fin:
                    reqs += [x.strip() for x in fin.read().splitlines()]
        for req in reqs:
            clean_req = req.replace('<', ' ').replace('>', ' ').replace(
                '=', ' ').split(' ')[0]
            self.version_map[clean_req] = req

    def check(self, check_pkgs):
        """
        install if the package is not importable.
        """
        for pkg in check_pkgs:
            if not _is_package_installed(pkg):
                logger.info(f'Installing {pkg} ...')
                if pkg in self.version_map:
                    pkg = self.version_map[pkg]
                pip_cmd = [sys.executable, '-m', 'pip', 'install', pkg]
                subprocess.check_call(pip_cmd)
                logger.info(f'The {pkg} installed.')
            if pkg == 'torch':
                _torch_check_and_set()
