import os

from data_juicer.format import MixtureFormatter, RemoteFormatter
from data_juicer.format.formatter import FORMATTERS, BaseFormatter
from data_juicer.utils.file_utils import (find_files_with_suffix,
                                          is_absolute_path)


def load_formatter(dataset_path,
                   generated_dataset_config=None,
                   text_keys=None,
                   suffixes=[],
                   add_suffix=False,
                   **kwargs) -> BaseFormatter:
    """
    Load mixture formatter for multiple different data formats with an optional
    weight(default 1.0) according to their formats.

    :param dataset_path: path to a dataset file or a dataset directory
    :param generated_dataset_config: Configuration used to create a dataset.
        The dataset will be created from this configuration if provided.
        It must contain the `type` field to specify the dataset name.
    :param text_keys: key names of field that stores sample text.
        Default: None
    :param suffixes: files with specified suffixes to be processed.
    :param add_suffix: whether to add the file suffix to dataset meta
        info
    :return: a dataset formatter.
    """
    if generated_dataset_config:
        assert isinstance(generated_dataset_config,
                          dict) and 'type' in generated_dataset_config
        args = generated_dataset_config.copy()
        obj_name = args.pop('type')
        args.update(kwargs)

        from .formatter import FORMATTERS
        return FORMATTERS.modules[obj_name](**args)

    formatter = MixtureFormatter(dataset_path=dataset_path,
                                 text_keys=text_keys,
                                 suffixes=suffixes,
                                 add_suffix=add_suffix,
                                 **kwargs)
    return formatter


def _load_formatter(dataset_path,
                    text_keys=None,
                    suffixes=None,
                    add_suffix=False,
                    **kwargs) -> BaseFormatter:
    """
    Load the appropriate formatter for different types of data formats.

    :param dataset_path: Path to dataset file or dataset directory
    :param text_keys: key names of field that stores sample text.
        Default: None
    :param suffixes: the suffix of files that will be read. Default:
        None
    :return: a dataset formatter.
    """

    if suffixes is None:
        suffixes = []
    ext_num = {}
    if os.path.isdir(dataset_path) or os.path.isfile(dataset_path):
        file_dict = find_files_with_suffix(dataset_path, suffixes)
        if not file_dict:
            raise IOError(
                'Unable to find files matching the suffix from {}'.format(
                    dataset_path))
        for ext in file_dict:
            ext_num[ext] = len(file_dict[ext])

    # local dataset
    if ext_num:
        formatter_num = {}
        for name, formatter in FORMATTERS.modules.items():
            formatter_num[name] = 0
            for ext in ext_num:
                if ext in formatter.SUFFIXES:
                    formatter_num[name] += ext_num[ext]
        formatter = max(formatter_num, key=lambda x: formatter_num[x])
        target_suffixes = set(ext_num.keys()).intersection(
            set(FORMATTERS.modules[formatter].SUFFIXES))
        return FORMATTERS.modules[formatter](dataset_path,
                                             text_keys=text_keys,
                                             suffixes=target_suffixes,
                                             add_suffix=add_suffix,
                                             **kwargs)

    # try huggingface dataset hub
    elif not is_absolute_path(dataset_path) and dataset_path.count('/') <= 1:
        return RemoteFormatter(dataset_path, text_keys=text_keys, **kwargs)

    # no data
    else:
        raise ValueError(f'Unable to load the dataset from [{dataset_path}]. '
                         f'It might be because Data-Juicer doesn\'t support '
                         f'the format of this dataset, or the path of this '
                         f'dataset is incorrect.Please check if it\'s a valid '
                         f'dataset path and retry.')
