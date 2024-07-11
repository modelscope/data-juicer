import sys
from functools import wraps

import pyarrow as pa
from loguru import logger

from data_juicer.utils.availability_utils import UNAVAILABLE_OPERATORS

from .base_op import OPERATORS, Mapper
from .op_fusion import fuse_operators


def convert_arrow_to_python(method):

    @wraps(method)
    def wrapper(sample, *args, **kwargs):
        if isinstance(sample, pa.Table):
            sample = sample.to_pydict()
        return method(sample, *args, **kwargs)

    return wrapper


def load_ops(process_list, op_fusion=False):
    """
    Load op list according to the process list from config file.

    :param process_list: A process list. Each item is an op name and its
        arguments.
    :param op_fusion: whether to fuse ops that share the same intermediate
        variables.
    :return: The op instance list.
    """
    ops = []
    new_process_list = []
    for process in process_list:
        op_name, args = list(process.items())[0]
        if op_name in UNAVAILABLE_OPERATORS:
            logger.error(UNAVAILABLE_OPERATORS[op_name].get_warning_msg())
            sys.exit(UNAVAILABLE_OPERATORS[op_name].get_warning_msg())
        op = OPERATORS.modules[op_name](**args)
        if isinstance(op, Mapper) and op.is_batched_op():
            op.process = convert_arrow_to_python(op.process)
        ops.append(op)
        new_process_list.append(process)

    # detect filter groups
    if op_fusion:
        new_process_list, ops = fuse_operators(new_process_list, ops)

    return new_process_list, ops
