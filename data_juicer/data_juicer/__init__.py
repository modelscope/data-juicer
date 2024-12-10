__version__ = '0.2.0'

import os
import subprocess
import sys

import multiprocess as mp
from loguru import logger

from data_juicer.utils.availability_utils import _is_package_available

# For now, only INFO will be shown. Later the severity level will be changed
# when setup_logger is called to initialize the logger.
logger.remove()
logger.add(sys.stderr, level='INFO')


def _cuda_device_count():
    _torch_available = _is_package_available('torch')

    if _torch_available:
        import torch
        return torch.cuda.device_count()

    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '-L'],
                                                    text=True)
        all_devices = nvidia_smi_output.strip().split('\n')

        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        if cuda_visible_devices is not None:
            logger.warning(
                'CUDA_VISIBLE_DEVICES is ignored when torch is unavailable. '
                'All detected GPUs will be used.')

        return len(all_devices)
    except Exception:
        # nvidia-smi not found or other error
        return 0


_USE_CUDA = False
_CUDA_COUNT = _cuda_device_count()


def use_cuda():
    return _USE_CUDA


def cuda_device_count():
    return _CUDA_COUNT


def setup_mp():
    method = os.getenv('MP_START_METHOD', 'auto').lower()
    if method == 'auto':
        if _CUDA_COUNT > 0:
            # forkserver is more lightweight
            method = ('forkserver' if 'forkserver'
                      in mp.get_all_start_methods() else 'spawn')
        else:
            method = 'fork'
    try:
        logger.info(f"Setting multiprocess start method to '{method}'.")
        mp.set_start_method(method, force=True)
    except RuntimeError as e:
        logger.warning(f'Error setting multiprocess start method: {e}')


def setup_cuda():
    global _USE_CUDA

    method = mp.get_start_method()
    if method != 'fork' and _CUDA_COUNT > 0:
        _USE_CUDA = True
    else:
        _USE_CUDA = False
    logger.debug(f'_USE_CUDA: {_USE_CUDA} | MP: {method} '
                 f'({mp.current_process().name})')


def initialize():
    if mp.current_process().name == 'MainProcess':
        setup_mp()
    setup_cuda()


initialize()
