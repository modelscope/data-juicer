import traceback
from collections import OrderedDict

import psutil
import pynvml

import os
from datetime import datetime
from monitor import MonitorWrapper

from data_juicer.config.config import global_cfg

dj_monitor = MonitorWrapper(mode=global_cfg.executor_type)


class DjError(Exception):
    """
        Super class of all DJ exception types, with enhanced exception chaining
    """

    def __init__(self,
                 err_message: str = None,
                 dj_ctx_data: str = None,
                 dj_ctx_op: str = None,
                 dj_ctx_runtime: str = None,
                 error_handle_hook=None,
                 info_collect_hook=None,
                 *args,
                 **kwargs):
        """
        Generally, once the DjError is raised, the error message will be logged
         to monitor, and then the exception will be auto-handled by the
         corresponding handlers

        :param err_message: basic error message of current exception
        :param dj_context_data: additional context information w.r.t.
            data-juicer's specific data sample
        :param dj_context_op: additional context information w.r.t.
            data-juicer's specific operators
        :param dj_context_runtime: additional context information w.r.t.
            data-juicer's specific processing runtime
        :param dj_handler: allowing custom exception handler
        """
        super().__init__(err_message)
        self.message = err_message
        self.dj_ctx_data = dj_ctx_data
        self.dj_ctx_op = dj_ctx_op
        self.dj_ctx_runtime = dj_ctx_runtime
        if error_handle_hook is not None:
            self.error_handle_hook = error_handle_hook
        if info_collect_hook is not None:
            self.info_collect_hook = info_collect_hook

        self.log_to_monitor(*args, **kwargs)
        # try to handle the error using the default
        # or passing (customized) handler
        try:
            self.error_handle_hook(*args, **kwargs)
        except Exception as e:
            print(
                f"Error occurred in DJ error handling when using default "
                f"or specified `error_handle_hook`: {e}\n")
            print(f"The `error_handle_hook` is: "
                  f"{self.error_handle_hook}\n")
            self.log_to_monitor()

    def error_handling_func(self, *args, **kwargs):
        """Custom exception handler"""

    def _collect_basic_info(self):
        basic_info = OrderedDict()
        # the order of the fields to be formatted (optional)
        # by default, show the `error_message` and `traceback_info`
        # in the beginning
        basic_info["top_fields"] = \
            ["error_message", "traceback_info"]

        # default fields
        basic_info["error_message"] = \
            f"The error message: {self.message}" if self.message else ""
        basic_info["traceback_info"] = \
            f"The traceback info: {traceback.format_exc()}"
        basic_info["dj_ctx_data"] = \
            f"The DJ context (data info): {self.dj_ctx_data}" \
                if self.dj_ctx_data else ""
        basic_info["dj_ctx_op"] = \
            f"The DJ context (operator info): {self.dj_ctx_op}" \
                if self.dj_ctx_op else ""
        basic_info["dj_ctx_runtime"] = \
            f"The DJ context (runtime info): {self.dj_ctx_runtime}" \
                if self.dj_ctx_runtime else ""

        # OS-related info
        current_pid = os.getpid()
        current_time = datetime.now()
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory()
        basic_info["current_pid"] = current_pid
        basic_info["current_time"] = current_time
        basic_info["cpu_usage"] = cpu_usage
        basic_info["memory_usage"] = memory_usage
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        avg_gpu_mem_usage = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            avg_gpu_mem_usage += memory_info.used
        basic_info[f"avg_gpu_mem_usage"] = \
            f"{avg_gpu_mem_usage / device_count / 1024 ** 2} MB"
        pynvml.nvmlShutdown()

        return basic_info

    def log_to_monitor(self, *args, **kwargs):
        """Log the error to monitor"""
        log_info = self._collect_basic_info()
        try:
            log_info = self.info_collect_hook(log_info, *args, **kwargs)
        except Exception as e:
            print(f"Error occurred in log_to_monitor when using specified "
                  f"`info_collecting_func`: {e}\n")
            print(f"The `info_collecting_func` is: "
                  f"{self.info_collect_hook}\n")

        dj_monitor.add_collected_log_info(log_info)

    def info_collecting_hook(self, basic_log_info, *args, **kwargs):
        # to be implemented by specific subclasses
        return basic_log_info

    def error_handling_hook(self, *args, **kwargs):
        # to be implemented by specific subclasses
        pass


class DjDatasetError(DjError):
    def info_collecting_hook(self, basic_log_info, *args, **kwargs):
        basic_log_info["error_message"] = \
            f"DjDatasetError occurred! " + \
            basic_log_info["error_message"]
        error_dataset = kwargs.get("error_data_sample")
        if error_dataset is not None:
            basic_log_info["dj_ctx_data"] = (basic_log_info["dj_ctx_data"]
                                             + str(error_dataset))
        basic_log_info["top_fields"] = ["error_message", "dj_ctx_data"]
        return basic_log_info

    def error_handling_hook(self, *args, **kwargs):
        # just skipping for the general DjDatasetError
        pass


class DjOpError(DjError):
    def info_collecting_hook(self, basic_log_info, *args, **kwargs):
        # to be implemented by specific subclasses
        basic_log_info["error_message"] = \
            f"DjOpError occurred! " + \
            basic_log_info["error_message"]
        basic_log_info["top_fields"] = ["error_message", "dj_ctx_op"]
        return basic_log_info

    def error_handling_hook(self, *args, **kwargs):
        # to be implemented by specific subclasses, e.g.,
        pass


class DjRuntimeError(DjError):
    def info_collecting_hook(self, basic_log_info, *args, **kwargs):
        basic_log_info["error_message"] = \
            f"DjRuntimeError occurred! " + \
            basic_log_info["error_message"]
        basic_log_info["top_fields"] = ["error_message", "dj_ctx_runtime",
                                        "current_pid", "current_time",
                                        "cpu_usage", "memory_usage",
                                        "avg_gpu_mem_usage"]
        # to be implemented by specific subclasses
        return basic_log_info

    def error_handling_hook(self, *args, **kwargs):
        # to be implemented by specific subclasses
        pass

#
# class TimeoutError(DjRuntimeError):
# class OPModelError(DjOpError):
# class OPNullFrameError(DjOpError):
#
