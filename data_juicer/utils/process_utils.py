import math
import subprocess

import psutil
from loguru import logger

from data_juicer import cuda_device_count, use_cuda


def get_min_cuda_memory():
    # get cuda memory info using "nvidia-smi" command
    import torch
    min_cuda_memory = torch.cuda.get_device_properties(
        0).total_memory / 1024**2
    nvidia_smi_output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'
    ]).decode('utf-8')
    for line in nvidia_smi_output.strip().split('\n'):
        free_memory = int(line)
        min_cuda_memory = min(min_cuda_memory, free_memory)
    return min_cuda_memory


def calculate_np(num_proc, op, op_name):
    """Calculate the optimum number of processes for the given OP"""
    if num_proc is None:
        num_proc = psutil.cpu_count()
    if use_cuda() and op._accelerator == 'cuda':
        cuda_mem_available = get_min_cuda_memory() / 1024
        op_proc = min(
            num_proc,
            math.floor(cuda_mem_available / (op.mem_required + 0.1)) *
            cuda_device_count())
        if use_cuda() and op.mem_required == 0:
            logger.warning(f'The required cuda memory of Op[{op_name}] '
                           f'has not been specified. '
                           f'Please specify the mem_required field in the '
                           f'config file, or you might encounter CUDA '
                           f'out of memory error. You can reference '
                           f'the mem_required field in the '
                           f'config_all.yaml file. ')
        if op_proc < 1.0:
            logger.warning(
                f'The required cuda memory:{op.mem_required}GB might '
                f'be more than the available cuda memory:'
                f'{cuda_mem_available}GB.'
                f'This Op [{op_name}] might '
                f'require more resource to run.')
        op_proc = max(op_proc, 1)
        return op_proc
    else:
        op_proc = num_proc
        cpu_available = psutil.cpu_count()
        mem_available = psutil.virtual_memory().available
        mem_available = mem_available / 1024**3
        op_proc = min(op_proc, math.floor(cpu_available / op.cpu_required))
        op_proc = min(op_proc,
                      math.floor(mem_available / (op.mem_required + 0.1)))
        if op_proc < 1.0:
            logger.warning(f'The required CPU number:{op.cpu_required} '
                           f'and memory:{op.mem_required}GB might '
                           f'be more than the available CPU:{cpu_available} '
                           f'and memory :{mem_available}GB.'
                           f'This Op [{op_name}] might '
                           f'require more resource to run.')
        op_proc = max(op_proc, 1)
        return op_proc
