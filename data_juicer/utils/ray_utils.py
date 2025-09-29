import os
import subprocess

import psutil
from loguru import logger

from data_juicer.utils.constant import RAY_JOB_ENV_VAR
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")

_RAY_NODES_INFO = None


def is_ray_mode():
    if int(os.environ.get(RAY_JOB_ENV_VAR, "0")):
        return True

    return False


def initialize_ray(cfg=None, force=False):
    if ray.is_initialized() and not force:
        return

    from ray.runtime_env import RuntimeEnv

    if cfg is None:
        ray_address = "auto"
        logger.warning("No ray config provided, using default ray address 'auto'.")
    else:
        ray_address = cfg.ray_address

    runtime_env = RuntimeEnv(env_vars={RAY_JOB_ENV_VAR: os.environ.get(RAY_JOB_ENV_VAR, "0")})
    ray.init(ray_address, ignore_reinit_error=True, runtime_env=runtime_env)


def check_and_initialize_ray(cfg=None):
    if is_ray_mode():
        initialize_ray(cfg)
        return True

    return False


def get_ray_nodes_info(cfg=None):
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
            "free_memory": free_mem,  # MB
            "cpu_count": cpu_count,
            "gpu_count": len(free_gpus_memory),
            "free_gpus_memory": free_gpus_memory,  # MB
        }

    initialize_ray(cfg)

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


def ray_cpu_count():
    cluster_resources = ray.cluster_resources()
    available_cpu = cluster_resources.get("CPU", 0)
    return available_cpu


def ray_gpu_count():
    cluster_resources = ray.cluster_resources()
    available_gpu = cluster_resources.get("GPU", 0)
    return available_gpu


def ray_available_memories():
    """Available memory for each alive node in MB."""
    ray_nodes_info = get_ray_nodes_info()

    available_mems = []
    for nodeid, info in ray_nodes_info.items():
        available_mems.append(info["free_memory"])

    return available_mems


def ray_available_gpu_memories():
    """Available gpu memory of each gpu card for each alive node in MB."""
    ray_nodes_info = get_ray_nodes_info()

    available_gpu_mems = []
    for nodeid, info in ray_nodes_info.items():
        available_gpu_mems.extend(info["free_gpus_memory"])

    return available_gpu_mems
