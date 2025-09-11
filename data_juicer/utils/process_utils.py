import math
import os
import subprocess

import multiprocess as mp
from loguru import logger

from data_juicer.utils.resource_utils import (
    available_gpu_memories,
    available_memories,
    cpu_count,
    cuda_device_count,
)


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


def calculate_np(name, mem_required, cpu_required, use_cuda=False, gpu_required=0):
    """Calculate the optimum number of processes for the given OP automatically。"""

    if not use_cuda and gpu_required > 0:
        raise ValueError(
            f"Op[{name}] attempted to request GPU resources (gpu_required={gpu_required}), "
            "but appears to lack GPU support. If you have verified this operator support GPU acceleration, "
            'please explicitly set its property: `_accelerator = "cuda"`.'
        )

    eps = 1e-9  # about 1 byte
    cpu_num = cpu_count()

    if use_cuda:
        cuda_mems_available = [m / 1024 for m in available_gpu_memories()]  # GB
        gpu_count = cuda_device_count()
        if not mem_required and not gpu_required:
            auto_num_proc = gpu_count
            logger.warning(
                f"The required cuda memory and gpu of Op[{name}] "
                f"has not been specified. "
                f"Please specify the mem_required field or gpu_required field in the "
                f"config file. You can reference the config_all.yaml file."
                f"Set the auto `num_proc` to number of GPUs {auto_num_proc}."
            )
        else:
            auto_proc_from_mem = sum(
                [math.floor(mem_available / (mem_required + eps)) for mem_available in cuda_mems_available]
            )
            auto_proc_from_gpu = math.floor(gpu_count / (gpu_required + eps))
            auto_proc_from_cpu = math.floor(cpu_num / (cpu_required + eps))
            auto_num_proc = min(auto_proc_from_mem, auto_proc_from_gpu, auto_proc_from_cpu)
            if auto_num_proc < 1:
                auto_num_proc = len(available_memories())  # set to the number of available nodes

            logger.info(
                f"Set the auto `num_proc` to {auto_num_proc} of Op[{name}] based on the "
                f"required cuda memory: {mem_required}GB "
                f"required gpu: {gpu_required} and required cpu: {cpu_required}."
            )
        return auto_num_proc
    else:
        mems_available = [m / 1024 for m in available_memories()]  # GB
        auto_proc_from_mem = sum([math.floor(mem_available / (mem_required + eps)) for mem_available in mems_available])
        auto_proc_from_cpu = math.floor(cpu_num / (cpu_required + eps))

        auto_num_proc = min(cpu_num, auto_proc_from_mem, auto_proc_from_cpu)

        if auto_num_proc < 1.0:
            auto_num_proc = len(available_memories())  # number of processes is equal to the number of nodes
            logger.warning(
                f"The required CPU number: {cpu_required} "
                f"and memory: {mem_required}GB might "
                f"be more than the available CPU: {cpu_num} "
                f"and memory: {mems_available}GB."
                f"This Op [{name}] might "
                f"require more resource to run. "
                f"Set the auto `num_proc` to available nodes number {auto_num_proc}."
            )
        else:
            logger.info(
                f"Set the auto `num_proc` to {auto_num_proc} of Op[{name}] based on the "
                f"required memory: {mem_required}GB "
                f"and required cpu: {cpu_required}."
            )
        return auto_num_proc
