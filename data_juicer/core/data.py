import copy
import inspect
from functools import wraps
from typing import Union

from datasets import Dataset, DatasetDict, is_caching_enabled
from datasets.formatting.formatting import LazyBatch
from loguru import logger

from data_juicer.utils import cache_utils
from data_juicer.utils.compress import (cleanup_compressed_cache_files,
                                        compress, decompress, CompressionOff)
from data_juicer.utils.fingerprint_utils import generate_fingerprint


def wrap_func_with_nested_access(f):
    """
    Before conducting actual function `f`, wrap its args and kargs into nested
    ones.

    :param f: function to be wrapped.
    :return: wrapped function
    """

    def wrap_nested_structure(*args, **kargs):
        wrapped_args = [nested_obj_factory(arg) for arg in args]
        wrapped_kargs = {
            k: nested_obj_factory(arg)
            for k, arg in kargs.items()
        }
        return wrapped_args, nested_obj_factory(wrapped_kargs)

    @wraps(f)
    def wrapped_f(*args, **kargs):
        args, kargs = wrap_nested_structure(*args, **kargs)
        # to ensure the args passing to the final calling of f can be nested,
        # in case of deeper-order wrapper funcs de-wrap this nesting behavior
        args = [
            wrap_func_with_nested_access(arg) if callable(arg) else arg
            for arg in args
        ]
        kargs = {
            k: (wrap_func_with_nested_access(arg) if callable(arg) else arg)
            for (k, arg) in kargs.items()
        }
        return f(*args, **kargs)

    return wrapped_f


def nested_obj_factory(obj):
    """
    Use nested classes to wrap the input object.

    :param obj: object to be nested.
    :return: nested object
    """
    if isinstance(obj, Dataset):
        return NestedDataset(obj)
    elif isinstance(obj, DatasetDict):
        return NestedDatasetDict(obj)
    elif isinstance(obj, dict):
        return NestedQueryDict(obj)
    elif isinstance(obj, LazyBatch):
        obj.data = NestedQueryDict(obj.data)
        return obj
    elif isinstance(obj, list):
        return [nested_obj_factory(item) for item in obj]
    else:
        return obj


class NestedQueryDict(dict):
    """Enhanced dict for better usability."""

    def __init__(self, *args, **kargs):
        if len(args) == 1 and isinstance(args[0], Dataset):
            # init from another DatasetDict instance
            self.__dict__ = copy.copy(args[0].__dict__)
        else:
            # init from scratch
            super().__init__(*args, **kargs)

        # batched sample, (k & v) are organized by list manner
        for k, v in self.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                self[k] = [NestedQueryDict(item) for item in v]

    def __getitem__(self, key):
        return nested_query(self, key)


class NestedDatasetDict(DatasetDict):
    """Enhanced HuggingFace-DatasetDict for better usability and efficiency."""

    def __init__(self, *args, **kargs):
        if len(args) == 1 and isinstance(args[0], Dataset):
            # init from another DatasetDict instance
            self.__dict__ = copy.copy(args[0].__dict__)
        else:
            # init from scratch
            super().__init__(*args, **kargs)

    def __getitem__(self, key):
        return nested_query(self, key)

    def map(self, **args):
        """Override the map func, which is called by most common operations,
        such that the processed samples can be accessed by nested manner."""
        if 'function' not in args or args['function'] is None:
            args['function'] = lambda x: nested_obj_factory(x)
        else:
            args['function'] = wrap_func_with_nested_access(args['function'])

        return super().map(**args)


class NestedDataset(Dataset):
    """Enhanced HuggingFace-Dataset for better usability and efficiency."""

    def __init__(self, *args, **kargs):
        if len(args) == 1 and isinstance(args[0], Dataset):
            # init from another Dataset instance
            self.__dict__ = copy.copy(args[0].__dict__)
        else:
            # init from scratch
            super().__init__(*args, **kargs)

        self.need_to_cleanup_caches = not is_caching_enabled()

    def __getitem__(self, key):
        if isinstance(key, str):
            # to index columns by query as string name(s)
            res = nested_query(self, key)
        else:
            # to index rows by query as integer index, slices,
            # or iter of indices or bools
            res = super().__getitem__(key)
        return nested_obj_factory(res)

    def map(self, *args, **kargs):
        """Override the map func, which is called by most common operations,
        such that the processed samples can be accessed by nested manner."""
        if args:
            args = list(args)
            # the first positional para is function
            if args[0] is None:
                args[0] = lambda x: nested_obj_factory(x)
            else:
                args[0] = wrap_func_with_nested_access(args[0])
            called_func = args[0]
        else:
            if 'function' not in kargs or kargs['function'] is None:
                kargs['function'] = lambda x: nested_obj_factory(x)
            else:
                kargs['function'] = wrap_func_with_nested_access(
                    kargs['function'])
            called_func = kargs['function']

        # For wrapped function, try to get its original unwrapped method
        while hasattr(called_func, '__wrapped__'):
            called_func = called_func.__wrapped__
        # Does the called function belong to a batched OP?
        if inspect.ismethod(called_func) \
                and 'is_batched_op' in dir(called_func.__self__) \
                and callable(getattr(called_func.__self__, 'is_batched_op')) \
                and called_func.__self__.is_batched_op():
            kargs['batched'] = True
            kargs['batch_size'] = 1

        if 'new_fingerprint' not in kargs or kargs['new_fingerprint'] is None:
            new_fingerprint = generate_fingerprint(self, *args, **kargs)
            kargs['new_fingerprint'] = new_fingerprint

        if cache_utils.CACHE_COMPRESS:
            decompress(self,
                       kargs['new_fingerprint'],
                       kargs['num_proc'] if 'num_proc' in kargs else 1)

        new_ds = NestedDataset(super().map(*args, **kargs))

        if cache_utils.CACHE_COMPRESS:
            compress(self,
                     new_ds,
                     kargs['num_proc'] if 'num_proc' in kargs else 1)

        if self.need_to_cleanup_caches:
            new_ds.cleanup_cache_files()

        return new_ds

    def filter(self, *args, **kargs):
        """Override the filter func, which is called by most common operations,
        such that the processed samples can be accessed by nested manner."""
        if args:
            args = list(args)
            # the first positional para is function
            if args[0] is None:
                args[0] = lambda x: nested_obj_factory(x)
            else:
                args[0] = wrap_func_with_nested_access(args[0])
        else:
            if 'function' not in kargs or kargs['function'] is None:
                kargs['function'] = lambda x: nested_obj_factory(x)
            else:
                kargs['function'] = wrap_func_with_nested_access(
                    kargs['function'])

        if 'new_fingerprint' not in kargs or kargs['new_fingerprint'] is None:
            new_fingerprint = generate_fingerprint(self, *args, **kargs)
            kargs['new_fingerprint'] = new_fingerprint

        # For filter, it involves a map and a filter operations, so the final
        # cache files includes two sets with different fingerprint (before and
        # after). So we need to decompress these two sets of compressed cache
        # files
        if cache_utils.CACHE_COMPRESS:
            decompress(self,
                       [kargs['new_fingerprint'], self._fingerprint],
                       kargs['num_proc'] if 'num_proc' in kargs else 1)

        # Turn off the compression due to it invokes map actually in the filter
        # function. For cache file changes, map: A -> B, filter: A -> A, B. If
        # we compress the caches of map, ops after filter cannot find the cache
        # files A. So we turn off the inner cache compression for filter.
        # Same for cleaning up cache files.
        with CompressionOff():
            prev_state = self.need_to_cleanup_caches
            self.need_to_cleanup_caches = False
            new_ds = NestedDataset(super().filter(*args, **kargs))
            self.need_to_cleanup_caches = prev_state

        if cache_utils.CACHE_COMPRESS:
            compress(self,
                     new_ds,
                     kargs['num_proc'] if 'num_proc' in kargs else 1)

        if self.need_to_cleanup_caches:
            new_ds.cleanup_cache_files()

        return new_ds

    def select(self, *args, **kargs):
        """Override the select func, such that selected samples can be accessed
        by nested manner."""
        return nested_obj_factory(super().select(*args, **kargs))

    @classmethod
    def from_dict(cls, *args, **kargs):
        """Override the from_dict func, which is called by most from_xx
        constructors, such that the constructed dataset object is
        NestedDataset."""
        return NestedDataset(super().from_dict(*args, **kargs))

    def add_column(self, *args, **kargs):
        """Override the add column func, such that the processed samples
        can be accessed by nested manner."""
        return NestedDataset(super().add_column(*args, **kargs))

    def select_columns(self, *args, **kargs):
        """Override the select columns func, such that the processed samples
        can be accessed by nested manner."""
        return NestedDataset(super().select_columns(*args, **kargs))

    def remove_columns(self, *args, **kargs):
        """Override the remove columns func, such that the processed samples
        can be accessed by nested manner."""
        return NestedDataset(super().remove_columns(*args, **kargs))

    def cleanup_cache_files(self):
        """Override the cleanup_cache_files func, clear raw and compressed
        cache files."""
        cleanup_compressed_cache_files(self)
        return super().cleanup_cache_files()


def nested_query(root_obj: Union[NestedDatasetDict, NestedDataset,
                                 NestedQueryDict], key):
    """
    Find item from a given object, by first checking flatten layer, then
    checking nested layers.

    :param root_obj: the object
    :param key: the stored item to be queried, e.g., "meta" or
        "meta.date"
    :return:
    """
    subkeys = key.split('.')

    tmp = root_obj
    for i in range(len(subkeys)):
        try:
            key_to_query = '.'.join(subkeys[i:len(subkeys)])
            if isinstance(tmp,
                          (NestedQueryDict, NestedDataset, NestedDatasetDict)):
                # access field using base_class's func to avoid endless loop
                res = super(type(tmp), tmp).__getitem__(key_to_query)
            elif isinstance(tmp, list):
                # NestedDataset may return multiple rows as list
                res = [nested_query(item, key_to_query) for item in tmp]
            else:
                # NestedQueryDict may return single row
                res = tmp[key_to_query]
            if res is not None:
                return res
        except Exception as outer_get_error:
            exist_in_dict = issubclass(type(tmp), dict) and \
                                '.'.join(subkeys[i:i + 1]) in tmp
            exist_in_dataset = issubclass(type(tmp), Dataset) and '.'.join(
                subkeys[i:i + 1]) in tmp.features
            if exist_in_dict or exist_in_dataset:
                # dive into next level
                tmp = nested_obj_factory(tmp['.'.join(subkeys[i:i + 1])])
            else:
                logger.debug(
                    f'cannot find item given key={key} in dataset='
                    f'{root_obj}. For the final caught outer-exception,'
                    f'type is: {type(outer_get_error)}, '
                    f'info is: {outer_get_error}')
                return None

    return None
