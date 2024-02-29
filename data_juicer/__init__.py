__version__ = '0.1.3'

import os
import subprocess

import multiprocess as mp
from loguru import logger


def _cuda_device_count():
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '-L'],
                                                    text=True)
        all_devices = nvidia_smi_output.strip().split('\n')

        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')

        if cuda_visible_devices is not None:
            visible_devices = cuda_visible_devices.split(',')
            visible_devices = [int(dev.strip()) for dev in visible_devices]
            num_visible_devices = sum(1 for dev in visible_devices
                                      if 0 <= dev < len(all_devices))
        else:
            num_visible_devices = len(all_devices)

        return num_visible_devices
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
