import math
import os
import subprocess

import multiprocess as mp
import psutil
from loguru import logger

from data_juicer import cuda_device_count


def setup_mp(method=None):
    if mp.current_process().name != "MainProcess":
        return

    if method is None:
        method = ["fork", "forkserver", "spawn"]
    if not isinstance(method, (list, tuple)):
        method = [method]
    method = [m.lower() for m in method]

    env_method = os.getenv("MP_START_METHOD", "").lower()
    if env_method in method:
        method = [env_method]

    available_methods = mp.get_all_start_methods()
    for m in method:
        if m in available_methods:
            try:
                logger.debug(f"Setting multiprocess start method to '{m}'")
                mp.set_start_method(m, force=True)
            except RuntimeError as e:
                logger.warning(f"Error setting multiprocess start method: {e}")
            break


def get_min_cuda_memory():
    # get cuda memory info using "nvidia-smi" command
    import torch

    min_cuda_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
    nvidia_smi_output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    ).decode("utf-8")
    for line in nvidia_smi_output.strip().split("\n"):
        free_memory = int(line)
        min_cuda_memory = min(min_cuda_memory, free_memory)
    return min_cuda_memory


def calculate_np(name, mem_required, cpu_required, num_proc=None, use_cuda=False):
    """Calculate the optimum number of processes for the given OP"""
    eps = 1e-9  # about 1 byte

    if use_cuda:
        auto_num_proc = None
        cuda_mem_available = get_min_cuda_memory() / 1024
        if mem_required == 0:
            logger.warning(
                f"The required cuda memory of Op[{name}] "
                f"has not been specified. "
                f"Please specify the mem_required field in the "
                f"config file, or you might encounter CUDA "
                f"out of memory error. You can reference "
                f"the mem_required field in the "
                f"config_all.yaml file."
            )
        else:
            auto_num_proc = math.floor(cuda_mem_available / mem_required) * cuda_device_count()
            if cuda_mem_available / mem_required < 1.0:
                logger.warning(
                    f"The required cuda memory:{mem_required}GB might "
                    f"be more than the available cuda memory:"
                    f"{cuda_mem_available}GB."
                    f"This Op[{name}] might "
                    f"require more resource to run."
                )

        if auto_num_proc and num_proc:
            op_proc = min(auto_num_proc, num_proc)
            if num_proc > auto_num_proc:
                logger.warning(
                    f"The given num_proc: {num_proc} is greater than "
                    f"the value {auto_num_proc} auto calculated based "
                    f"on the mem_required of Op[{name}]. "
                    f"Set the `num_proc` to {auto_num_proc}."
                )
        elif not auto_num_proc and not num_proc:
            op_proc = cuda_device_count()
            logger.warning(
                f"Both mem_required and num_proc of Op[{name}] are not set."
                f"Set the `num_proc` to number of GPUs {op_proc}."
            )
        else:
            op_proc = auto_num_proc if auto_num_proc else num_proc

        op_proc = max(op_proc, 1)
        return op_proc
    else:
        if num_proc is None:
            num_proc = psutil.cpu_count()

        op_proc = num_proc
        cpu_available = psutil.cpu_count()
        mem_available = psutil.virtual_memory().available
        mem_available = mem_available / 1024**3
        op_proc = min(op_proc, math.floor(cpu_available / cpu_required + eps))
        op_proc = min(op_proc, math.floor(mem_available / (mem_required + eps)))
        if op_proc < 1.0:
            logger.warning(
                f"The required CPU number:{cpu_required} "
                f"and memory:{mem_required}GB might "
                f"be more than the available CPU:{cpu_available} "
                f"and memory :{mem_available}GB."
                f"This Op [{name}] might "
                f"require more resource to run."
            )
        op_proc = max(op_proc, 1)
        return op_proc
