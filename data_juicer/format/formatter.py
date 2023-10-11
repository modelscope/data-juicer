import os
from typing import List, Tuple, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from loguru import logger

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import (find_files_with_suffix,
                                          is_absolute_path)
from data_juicer.utils.registry import Registry

FORMATTERS = Registry('Formatters')


class BaseFormatter:
    """Base class to load dataset."""

    def load_dataset(self, *args) -> Dataset:
        raise NotImplementedError


class LocalFormatter(BaseFormatter):
    """The class is used to load a dataset from local files or local
    directory."""

    def __init__(
        self,
        dataset_path: str,
        type: str,
        suffixes: Union[str, List[str], Tuple[str]] = None,
        text_keys: List[str] = None,
        add_suffix=False,
        **kwargs,
    ):
        """
        Initialization method.

        :param dataset_path: path to a dataset file or a dataset
            directory
        :param type: a packaged dataset module type (json, csv, etc.)
        :param suffixes: files with specified suffixes to be processed
        :param text_keys: key names of field that stores sample
            text.
        :param add_suffix: whether to add the file suffix to dataset
            meta info
        :param kwargs: extra args
        """
        self.type = type
        self.kwargs = kwargs
        self.text_keys = text_keys
        self.data_files = find_files_with_suffix(dataset_path, suffixes)
        self.add_suffix = add_suffix

    def load_dataset(self, num_proc: int = 1) -> Dataset:
        """
        Load a dataset from dataset file or dataset directory, and unify its
        format.

        :param num_proc: number of processes when loading the dataset
        :param global_cfg: global cfg used in consequent processes,
        :return: formatted dataset
        """
        datasets = load_dataset(self.type,
                                data_files={
                                    key.strip('.'): self.data_files[key]
                                    for key in self.data_files
                                },
                                num_proc=num_proc,
                                **self.kwargs)
        if self.add_suffix:
            logger.info('Add suffix info into dataset...')
            datasets = add_suffixes(datasets)
        else:
            from data_juicer.core.data import NestedDataset
            datasets = NestedDataset(
                concatenate_datasets([ds for _, ds in datasets.items()]))
        ds = unify_format(datasets,
                          text_keys=self.text_keys,
                          num_proc=num_proc)
        return ds


class RemoteFormatter(BaseFormatter):
    """The class is used to load a dataset from repository of huggingface
    hub."""

    def __init__(self,
                 dataset_path: str,
                 text_keys: List[str] = None,
                 **kwargs):
        """
        Initialization method.

        :param dataset_path: a dataset file or a dataset directory
        :param text_keys: key names of field that stores sample
            text.
        :param kwargs: extra args
        """
        self.path = dataset_path
        self.text_keys = text_keys
        self.kwargs = kwargs

    def load_dataset(self, num_proc: int = 1) -> Dataset:
        """
        Load a dataset from HuggingFace, and unify its format.

        :param num_proc: number of processes when loading the dataset
        :param global_cfg: the global cfg used in consequent processes,
        :return: formatted dataset
        """
        ds = load_dataset(self.path,
                          split='train',
                          num_proc=num_proc,
                          **self.kwargs)
        ds = unify_format(ds, text_keys=self.text_keys, num_proc=num_proc)
        return ds


def add_suffixes(datasets: DatasetDict) -> Dataset:
    """
    Add suffix filed to datasets.

    :param datasets: a DatasetDict object
    :return: datasets with suffix features.
    """
    logger.info('Add suffix column for dataset')
    for key, ds in datasets.items():
        if Fields.suffix not in ds.features:
            datasets[key] = ds.add_column(name=Fields.suffix,
                                          column=['.' + key] * ds.num_rows)
    datasets = concatenate_datasets([ds for _, ds in datasets.items()])
    from data_juicer.core.data import NestedDataset
    return NestedDataset(datasets)


def unify_format(
    dataset: Dataset,
    text_keys: Union[List[str], str] = 'text',
    num_proc: int = 1,
) -> Dataset:
    """
    Get an unified internal format, conduct the following modifications.

    1. check keys of dataset

    2. filter out those samples with empty or None text

    :param dataset: input dataset
    :param text_keys: original text key(s) of dataset.
    :param num_proc: number of processes for mapping
    :param global_cfg: the global cfg used in consequent processes,
        since cfg.text_key may be modified after unifying

    :return: unified_format_dataset
    """
    from data_juicer.core.data import NestedDataset
    if isinstance(dataset, DatasetDict):
        datasets = list(dataset.values())
        assert len(datasets) == 1, 'Please make sure the passed datasets ' \
                                   'contains only 1 dataset'
        dataset = datasets[0]
    assert isinstance(dataset, Dataset) or \
           isinstance(dataset, NestedDataset), \
           'Currently we only support processing data' \
           'with huggingface-Dataset format'

    if text_keys is None:
        text_keys = []

    if isinstance(text_keys, str):
        text_keys = [text_keys]

    logger.info('Unifying the input dataset formats...')

    dataset = NestedDataset(dataset)

    # 1. check text related keys
    for key in text_keys:
        if key not in dataset.features:
            err_msg = f'There is no key [{key}] in dataset. You might set ' \
                      f'wrong text_key in the config file for your dataset. ' \
                      f'Please check and retry!'
            logger.error(err_msg)
            raise ValueError(err_msg)

    # 2. filter out those samples with empty or None text
    # TODO: optimize the filtering operation for better efficiency
    logger.info(f'There are {len(dataset)} sample(s) in the original dataset.')

    def non_empty_text(sample, target_keys):
        for target_key in target_keys:
            # TODO: case for CFT, in which the len(sample[target_key]) == 0
            if sample[target_key] is None:
                # we filter out the samples contains at least None column
                # since the op can not handle it now
                return False
        return True

    dataset = dataset.filter(non_empty_text,
                             num_proc=num_proc,
                             fn_kwargs={'target_keys': text_keys})
    logger.info(f'{len(dataset)} samples left after filtering empty text.')

    # 3. add Fields.stats field
    # TODO:
    # this is a temp solution,
    # it will occur errors when only call mapper ops
    # dataset = dataset.add_column( \
    # name=Fields.stats, column=[{}] * dataset.num_rows)

    return dataset


def load_formatter(dataset_path,
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
        raise NotImplementedError
