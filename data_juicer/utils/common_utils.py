import hashlib
import sys

import numpy as np
from loguru import logger


def stats_to_number(s, reverse=True):
    '''
        convert a stats value which can be string
        of list to a float.
    '''
    try:
        if isinstance(s, str):
            return float(s)
        if s is None or s == []:
            raise ValueError('empty value')
        return float(np.asarray(s).mean())
    except Exception:
        if reverse:
            return -sys.maxsize
        else:
            return sys.maxsize


def dict_to_hash(input_dict: dict, hash_length=None):
    """
        hash a dict to a string with length hash_length

        :param input_dict: the given dict
    """
    sorted_items = sorted(input_dict.items())
    dict_string = str(sorted_items).encode()
    hasher = hashlib.sha256()
    hasher.update(dict_string)
    hash_value = hasher.hexdigest()
    if hash_length:
        hash_value = hash_value[:hash_length]
    return hash_value


def get_val_by_nested_key(input_dict: dict, nested_key: str):
    """
        return val of the dict in the nested key.

        :param nested_key: the nested key, such as "__dj__stats__.text_len"
    """
    keys = nested_key.split('.')
    cur = input_dict
    for key in keys:
        if key not in cur:
            logger.warning(f'Unvisitable nested key: {nested_key}!')
        cur = cur[key]
    return cur
