__version__ = '0.1.3'

import os

import multiprocess as mp
from loguru import logger

__USE_CUDA = False
__CUDA_COUNT = 0


def use_cuda():
    return __USE_CUDA


def cuda_device_count():
    return __CUDA_COUNT


def setup_mp():
    method = os.getenv('MP_START_METHOD', 'auto').lower()
    if method == 'auto':
        import torch
        if torch.cuda.is_available():
            # forkserver is more lightweight
            method = 'forkserver' if 'forkserver' in mp.get_all_start_methods(
            ) else 'spawn'
        else:
            method = 'fork'
    try:
        logger.info(f"Setting multiprocess start method to '{method}'.")
        mp.set_start_method(method, force=True)
    except RuntimeError as e:
        logger.warning(f'Error setting multiprocess start method: {e}')


def setup_cuda():
    global __USE_CUDA, __CUDA_COUNT
    method = mp.get_start_method()
    import torch
    if method != 'fork' and torch.cuda.is_available():
        __USE_CUDA = True
    else:
        __USE_CUDA = False
    logger.debug(f'__USE_CUDA: {__USE_CUDA} | MP: {method} '
                 f'({mp.current_process().name})')

    if __USE_CUDA:
        __CUDA_COUNT = torch.cuda.device_count()


def initialize():
    if mp.current_process().name == 'MainProcess':
        setup_mp()
    setup_cuda()


initialize()
