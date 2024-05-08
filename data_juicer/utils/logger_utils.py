# Some codes here are adapted from
# https://github.com/MegEngine/YOLOX/blob/main/yolox/utils/logger.py

# Copyright 2021 Megvii, Base Detection
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import inspect
import os
import sys
from io import StringIO

from loguru import logger
from loguru._file_sink import FileSink

LOGGER_SETUP = False


def get_caller_name(depth=0):
    """
    Get caller name by depth.

    :param depth: depth of caller context, use 0 for caller depth.
    :return: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals['__name__']


class StreamToLoguru:
    """Stream object that redirects writes to a logger instance."""

    def __init__(self, level='INFO', caller_names=('datasets', 'logging')):
        """
        Initialization method.

        :param level: log level string of loguru. Default value: "INFO".
        :param caller_names: caller names of redirected module.
                    Default value: (apex, pycocotools).
        """
        self.level = level
        self.caller_names = caller_names
        self.buffer = StringIO()
        self.BUFFER_SIZE = 1024 * 1024

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit('.', maxsplit=-1)[0]
        self.buffer.write(buf)
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            # sys.__stdout__.write(buf)
            logger.opt(raw=True).info(buf)

        self.buffer.truncate(self.BUFFER_SIZE)

    def getvalue(self):
        return self.buffer.getvalue()

    def flush(self):
        self.buffer.flush()


def redirect_sys_output(log_level='INFO'):
    """
    Redirect stdout/stderr to loguru with log level.

    :param log_level: log level string of loguru. Default value: "INFO".
    """
    redirect_logger = StreamToLoguru(level=log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def get_log_file_path():
    """
    Get the path to the location of the log file.

    :return: a location of log file.
    """
    for _, handler in logger._core.handlers.items():
        if isinstance(handler._sink, FileSink):
            return handler._sink._file.name


def setup_logger(save_dir,
                 distributed_rank=0,
                 filename='log.txt',
                 mode='o',
                 level='INFO',
                 redirect=True):
    """
    Setup logger for training and testing.

    :param save_dir: location to save log file
    :param distributed_rank: device rank when multi-gpu environment
    :param filename: log file name to save
    :param mode: log file write mode, `append` or `override`. default is `o`.
    :param level: log severity level. It's "INFO" in default.
    :param redirect: whether to redirect system output
    :return: logger instance.
    """
    global LOGGER_SETUP

    if LOGGER_SETUP:
        return

    loguru_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == 'o' and os.path.exists(save_file):
        os.remove(save_file)

    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level=level,
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    if redirect:
        redirect_sys_output(level)
    LOGGER_SETUP = True


class HiddenPrints:
    """Define a range that hide the outputs within this range."""

    def __enter__(self):
        """
        Store the original standard output and redirect the standard output to
        null when entering this range.
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the redirected standard output and restore it when exiting from
        this range.
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout
