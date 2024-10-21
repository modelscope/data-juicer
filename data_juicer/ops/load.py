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
    new_process_list = []
    for process in process_list:
        op_name, args = list(process.items())[0]
        ops.append(OPERATORS.modules[op_name](**args))
        new_process_list.append(process)

    # detect filter groups
    if op_fusion:
        new_process_list, ops = fuse_operators(new_process_list, ops)

    for op_cfg, op in zip(new_process_list, ops):
        op._op_cfg = op_cfg

    return ops
