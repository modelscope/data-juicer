import math
import os
import subprocess
from typing import List

import multiprocess as mp
import psutil
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.utils.constant import RAY_JOB_ENV_VAR
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")
_RAY_NODES_INFO = None


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


def is_ray_initialized():
    try:
        return ray.is_initialized()
    except Exception:
        return False


def is_ray_mode():
    if int(os.environ.get(RAY_JOB_ENV_VAR, "0")):
        assert is_ray_initialized(), "Ray cluster is not initialized."
        return True

    return False


def cpu_count():
    if is_ray_mode():
        available_resources = ray.available_resources()
        available_cpu = available_resources.get("CPU", 0)
        return available_cpu

    return psutil.cpu_count()


def available_memories() -> List[int]:
    """Available memory for each node in MB."""
    if is_ray_mode():
        ray_nodes_info = get_ray_nodes_info()

        available_mems = []
        for nodeid, info in ray_nodes_info.items():
            available_mems.append(info["memory"])

        return available_mems

    return [int(psutil.virtual_memory().available / (1024**2))]


def available_gpu_memories() -> List[int]:
    """Available gpu memory of each gpu card for each alive node in MB."""
    if is_ray_mode():
        ray_nodes_info = get_ray_nodes_info()

        available_gpu_mems = []
        for nodeid, info in ray_nodes_info.items():
            available_gpu_mems.extend(info["gpus_memory"])

        return available_gpu_mems

    try:
        nvidia_smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).decode("utf-8")

        return [int(i) for i in nvidia_smi_output.strip().split("\n")]
    except Exception:
        return []


def get_ray_nodes_info():
    global _RAY_NODES_INFO

    if _RAY_NODES_INFO is not None:
        return _RAY_NODES_INFO

    @ray.remote
    def collect_node_info():
        mem_info = psutil.virtual_memory()
        free_mem = int(mem_info.available / (1024**2))  # MB
        cpu_count = psutil.cpu_count()

        try:
            free_gpus_memory = []
            nvidia_smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
            ).decode("utf-8")

            for line in nvidia_smi_output.strip().split("\n"):
                free_gpus_memory.append(int(line))

        except Exception:
            # no gpu
            free_gpus_memory = []

        return {
            "memory": free_mem,  # MB
            "cpu_count": cpu_count,
            "gpus_memory": free_gpus_memory,  # MB
        }

    assert is_ray_initialized(), "Ray cluster is not initialized."
    ray.init(address="auto", ignore_reinit_error=True)

    nodes = ray.nodes()
    alive_nodes = [node for node in nodes if node["Alive"]]
    # skip head node
    worker_nodes = [node for node in alive_nodes if "head" not in node["NodeManagerHostname"]]

    futures = []
    for node in worker_nodes:
        node_id = node["NodeID"]
        from ray.util import scheduling_strategies

        strategy = scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        future = collect_node_info.options(scheduling_strategy=strategy).remote()
        futures.append(future)

    results = ray.get(futures)

    _RAY_NODES_INFO = {}
    for i, (node, info) in enumerate(zip(alive_nodes, results)):
        node_id = node["NodeID"]
        _RAY_NODES_INFO[node_id] = info

    logger.info(f"Ray cluster info:\n{_RAY_NODES_INFO}")

    return _RAY_NODES_INFO


def calculate_np(name, mem_required, cpu_required, num_proc=None, use_cuda=False):
    """Calculate the optimum number of processes for the given OP"""
    eps = 1e-9  # about 1 byte

    if use_cuda:
        auto_num_proc = None
        cuda_mems_available = [m / 1024 for m in available_gpu_memories()]  # GB
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
            auto_num_proc = sum(
                [math.floor(cuda_mem_available / mem_required) for cuda_mem_available in cuda_mems_available]
            )
            if auto_num_proc < cuda_device_count():
                logger.warning(
                    f"The required cuda memory:{mem_required}GB might "
                    f"be more than the available cuda devices memory list:"
                    f"{cuda_mems_available}GB."
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
        elif auto_num_proc is None and num_proc is None:
            op_proc = cuda_device_count()
            logger.warning(
                f"Both mem_required and num_proc of Op[{name}] are not set."
                f"Set the `num_proc` to number of GPUs {op_proc}."
            )
        else:
            op_proc = auto_num_proc if auto_num_proc is not None else num_proc

        if op_proc <= 1:
            op_proc = len(available_memories())  # number of processes is equal to the number of nodes
        return op_proc
    else:
        cpu_num = cpu_count()
        if num_proc is None:
            num_proc = cpu_num

        op_proc = num_proc
        mems_available = [m / 1024 for m in available_memories()]  # GB
        auto_proc = sum([math.floor(mem_available / (mem_required + eps)) for mem_available in mems_available])
        op_proc = min(op_proc, auto_proc)

        if op_proc < 1.0:
            logger.warning(
                f"The required CPU number:{cpu_required} "
                f"and memory:{mem_required}GB might "
                f"be more than the available CPU:{cpu_num} "
                f"and memory :{mems_available}GB."
                f"This Op [{name}] might "
                f"require more resource to run."
            )
        if op_proc <= 1:
            op_proc = len(available_memories())  # number of processes is equal to the number of nodes
        return op_proc
