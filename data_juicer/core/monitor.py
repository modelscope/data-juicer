from datetime import datetime

import ray
import json
import wandb
from loguru import logger
from collections import OrderedDict


def format_log_info(log_info):
    """
        format log info (dict) to a whole string,
        according to the order of ordered_fields (optional)
    """
    non_null_field = OrderedDict()
    ordered_fields = log_info.get("ordered_fields", None)

    if not ordered_fields:
        for key in ordered_fields:
            if (key in log_info and
                    (log_info[key] is not None and log_info[key] != "")):
                non_null_field[key] = log_info[key]
    for k, v in log_info.items():
        if (v is None or v == "" or
                k in ordered_fields or k == "ordered_fields"):
            continue
        non_null_field[k] = v

    formatted_log_info = ""
    for k, v in non_null_field.items():
        formatted_log_info += f"{v}\n"
    return formatted_log_info


class MonitorWrapper(object):
    """
    collecting, summarizing and analyzing logs
    """

    def __init__(self, mode="default"):
        if mode not in ["default", "ray"]:
            raise ValueError("mode must be either standalone or ray")
        self.exec_mode = mode
        self.logger = logger
        if mode == "default":
            self.global_standalone_monitor = StandaloneMonitor()
            self.global_ray_monitor = None
        elif mode == "ray":
            self.global_ray_monitor = RayMonitor.remote()
            self.global_standalone_monitor = None

    def get_collected_log_info(self):
        if self.exec_mode == "default":
            return self.global_standalone_monitor.get_collected_log_info()
        elif self.exec_mode == "ray":
            return self.global_ray_monitor.get_collected_log_info.remote()

    def add_collected_log_info(self, log_info, level="info"):
        if level not in ["info", "warning", "error"]:
            raise ValueError("level must be either info, warning or error")
        log_info_str = format_log_info(log_info)
        if level == "info":
            return self.logger.info(log_info_str)
        elif level == "warning":
            return self.logger.warning(log_info_str)
        elif level == "error":
            return self.logger.error(log_info_str)

        if self.exec_mode == "default":
            self.global_standalone_monitor.add_collected_log_info(log_info)
        elif self.exec_mode == "ray":
            self.global_ray_monitor.add_collected_log_info.remote(log_info)

    def save_collected_log_info(self):
        if self.exec_mode == "default":
            self.global_standalone_monitor.save_collected_log_info()
        elif self.exec_mode == "ray":
            self.global_ray_monitor.save_collected_log_info.remote()


class StandaloneMonitor(object):
    """
    collecting, summarizing and analyzing logs
    """

    def __init__(self):
        self.collected_log_info = []
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.log_path = f"global_monitor_{timestamp}.log.json"

    def get_collected_log_info(self):
        return self.collected_log_info

    def add_collected_log_info(self, log_info):
        self.collected_log_info.append(log_info)

    def save_collected_log_info(self):
        """
        save collected log info to file
        """

        with open(self.log_path, "w") as f:
            json.dump(self.collected_log_info, f)
        if wandb.run is not None:
            wandb.log(self.collected_log_info)


@ray.remote
class RayMonitor(StandaloneMonitor):
    """
    global monitor for Ray, as Ray's drivers, tasks and actors
    don’t share the same address space.
    """

    def __init__(self):
        super(RayMonitor, self).__init__()
        self.ray_worker_monitor = {}

    def add_log(self, log_info):
        """
        add log info to monitor
        """
        if log_info["worker_ip"] not in self.ray_worker_monitor:
            self.ray_worker_monitor[log_info["worker_ip"]] = []
        self.ray_worker_monitor[log_info["worker_ip"]].append(log_info)

    def get_collected_log_info(self):
        """
        get collected log info
        """
        return self.ray_worker_monitor

    def save_collected_log_info(self):
        """
        save collected log info to file
        """
        for worker_ip, log_info in self.ray_worker_monitor.items():
            with open(f"worker_monitor_{worker_ip}.log.json", "w") as f:
                json.dump(log_info, f)
        super(RayMonitor, self).save_collected_log_info()
