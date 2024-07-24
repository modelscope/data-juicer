import sys

import numpy as np


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
