from .base_op import OPERATORS


def load_ops(process_list, text_key='text'):
    """
    Load op list according to the process list from config file.

    :param process_list: A process list. Each item is an op name and its
        arguments.
    :param text_key: the key name of field that stores sample texts to
        be processed.
    :return: The op instance list.
    """
    if isinstance(text_key, list):
        text_key = text_key[0]

    ops = []
    for process in process_list:
        op_name, args = list(process.items())[0]

        # users can freely specify text_key for different ops
        if args is None:
            args = {'text_key': text_key}
        elif args['text_key'] is None:
            args['text_key'] = text_key
        ops.append(OPERATORS.modules[op_name](**args))

    return ops
