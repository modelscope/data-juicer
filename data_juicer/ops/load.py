import sys

from loguru import logger

from data_juicer.utils.availability_utils import UNAVAILABLE_OPERATORS

from .base_op import OPERATORS


def load_ops(process_list):
    """
    Load op list according to the process list from config file.

    :param process_list: A process list. Each item is an op name and its
        arguments.
    :return: The op instance list.
    """
    ops = []
    new_process_list = []
    for process in process_list:
        op_name, args = list(process.items())[0]
        if op_name in UNAVAILABLE_OPERATORS:
            logger.error(UNAVAILABLE_OPERATORS[op_name].get_warning_msg())
            sys.exit(UNAVAILABLE_OPERATORS[op_name].get_warning_msg())
        ops.append(OPERATORS.modules[op_name](**args))
        new_process_list.append(process)

    # store the OP configs into each OP
    for op_cfg, op in zip(new_process_list, ops):
        op._op_cfg = op_cfg

    return ops
