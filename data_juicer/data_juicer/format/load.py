from .formatter import BaseFormatter
from .mixture_formatter import MixtureFormatter


def load_formatter(dataset_path,
                   text_keys=None,
                   suffixes=[],
                   add_suffix=False,
                   **kwargs) -> BaseFormatter:
    """
    Load mixture formatter for multiple different data formats with an optional
    weight(default 1.0) according to their formats.

    :param dataset_path: path to a dataset file or a dataset directory
    :param text_keys: key names of field that stores sample text.
        Default: None
    :param suffixes: files with specified suffixes to be processed.
    :param add_suffix: whether to add the file suffix to dataset meta
        info
    :return: a dataset formatter.
    """
    formatter = MixtureFormatter(dataset_path=dataset_path,
                                 text_keys=text_keys,
                                 suffixes=suffixes,
                                 add_suffix=add_suffix,
                                 **kwargs)
    return formatter
