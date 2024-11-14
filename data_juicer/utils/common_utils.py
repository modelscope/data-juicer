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
            return None
        cur = cur[key]
    return cur


def is_string_list(var):
    """
        return if the var is list of string.

        :param var: input variance
    """
    return isinstance(var, list) and all(isinstance(it, str) for it in var)


def avg_split_string_list_under_limit(str_list: list,
                                      token_nums: list,
                                      max_token_num=None):
    """
        Split the string list to several sub str_list, such that the total
        token num of each sub string list is less than max_token_num, keeping
        the total token nums of sub string lists are similar.

        :param str_list: input string list.
        :param token_nums: token num of each string list.
        :param max_token_num: max token num of each sub string list.
    """
    if max_token_num is None:
        return [str_list]

    if len(str_list) != len(token_nums):
        logger.warning('The length of str_list and token_nums must be equal!')
        return [str_list]

    total_num = sum(token_nums)
    if total_num <= max_token_num:
        return [str_list]

    group_num = total_num // max_token_num + 1
    avg_num = total_num / group_num
    res = []
    cur_list = []
    cur_sum = 0
    for text, token_num in zip(str_list, token_nums):
        if token_num > max_token_num:
            logger.warning(
                'Token num is greater than max_token_num in one sample!')
        if cur_sum + token_num > max_token_num and cur_list:
            res.append(cur_list)
            cur_list = []
            cur_sum = 0
        cur_list.append(text)
        cur_sum += token_num
        if cur_sum > avg_num:
            res.append(cur_list)
            cur_list = []
            cur_sum = 0
    if cur_list:
        res.append(cur_list)
    return res
