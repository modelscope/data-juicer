from .formatter import BaseFormatter
from .mixture_formatter import MixtureFormatter


def load_formatter(dataset_path,
                   keys_to_load=None,
                   suffixes=[],
                   add_suffix=False,
                   **kwargs) -> BaseFormatter:
    """
    Load mixture formatter for multiple different data formats with an optional
    weight(default 1.0) according to their formats.

    :param dataset_path: path to a dataset file or a dataset directory
    :param keys_to_load: key names of field that stores sample text.
        Default: ['text']
    :param suffixes: files with specified suffixes to be processed.
    :param add_suffix: whether to add the file suffix to dataset meta
        info
    :return: a dataset formatter.
    """
    if keys_to_load is None:
        keys_to_load = ['text']
    formatter = MixtureFormatter(dataset_path=dataset_path,
                                 keys_to_load=keys_to_load,
                                 suffixes=suffixes,
                                 add_suffix=add_suffix,
                                 **kwargs)
    return formatter
