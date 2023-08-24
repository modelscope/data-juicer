from .base_op import OPERATORS
from .op_fusion import fuse_operators

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
    for process in process_list:
        op_name, args = list(process.items())[0]
        ops.append(OPERATORS.modules[op_name](**args))

    # detect filter groups
    if op_fusion:
        process_list, ops = fuse_operators(process_list, ops)

    return process_list, ops
