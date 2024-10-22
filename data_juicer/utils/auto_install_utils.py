import os
import subprocess
import sys

from loguru import logger

from data_juicer.utils.auto_install_mapping import MODULE_TO_PKGS
from data_juicer.utils.availability_utils import _torch_check_and_set


def _is_module_installed(module_name):
    if module_name in MODULE_TO_PKGS:
        pkgs = MODULE_TO_PKGS[module_name]
    else:
        pkgs = [module_name]
    for pkg in pkgs:
        if not _is_package_installed(pkg):
            return False
    return True


def _is_package_installed(package_name):
    if '@' in package_name:
        package_name = package_name.split('@')[0]
    if '[' in package_name:
        package_name = package_name.split('[')[0]
    try:
        subprocess.check_output(
            [sys.executable, '-m', 'pip', 'show', '-q', package_name],
            stderr=subprocess.STDOUT)
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
                logger.warning(f'target file does not exist: {path}')
            else:
                with open(path, 'r', encoding='utf-8') as fin:
                    reqs += [x.strip() for x in fin.read().splitlines()]
        for req in reqs:
            clean_req = req.replace('<', ' ').replace('>', ' ').replace(
                '=', ' ').split(' ')[0]
            self.version_map[clean_req] = req

    def check(self, check_pkgs, param=None):
        """
        install if the package is not installed.

        :param check_pkgs: packages to be check, install them if they are
            not installed
        :param param: install param for pip if necessary
        """
        for pkg in check_pkgs:
            if not _is_package_installed(pkg):
                logger.warning(f'Installing {pkg} ...')
                if pkg in self.version_map:
                    pkg = self.version_map[pkg]
                # not install the dependency of this pkg
                if param is None:
                    pip_cmd = [sys.executable, '-m', 'pip', 'install', pkg]
                else:
                    pip_cmd = [
                        sys.executable, '-m', 'pip', 'install', param, pkg
                    ]
                subprocess.check_call(pip_cmd)
                logger.info(f'The {pkg} installed.')
            if pkg == 'torch':
                _torch_check_and_set()

    def install(self, module):
        """
        install package for given module.

        :param module: module to be installed
        """
        if module in MODULE_TO_PKGS:
            pkgs = MODULE_TO_PKGS[module]
        else:
            pkgs = [module]
        for pkg in pkgs:
            if pkg in self.version_map:
                pkg = self.version_map[pkg]
            pip_cmd = [sys.executable, '-m', 'pip', 'install', pkg]
            subprocess.check_call(pip_cmd)
            logger.info(f'The {pkg} installed.')
