import os
from typing import List, Tuple, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from jsonargparse import Namespace
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
        text_keys_to_load: List[str] = None,
        add_suffix=False,
        **kwargs,
    ):
        """
        Initialization method.

        :param dataset_path: path to a dataset file or a dataset
            directory
        :param type: a packaged dataset module type (json, csv, etc.)
        :param suffixes: files with specified suffixes to be processed
        :param text_keys_to_load: key names of field that stores sample
            text.
        :param add_suffix: whether to add the file suffix to dataset
            meta info
        :param kwargs: extra args
        """
        if text_keys_to_load is None:
            text_keys_to_load = ['text']
        self.type = type
        self.kwargs = kwargs
        self.text_keys_to_load = text_keys_to_load
        self.data_files = find_files_with_suffix(dataset_path, suffixes)
        self.add_suffix = add_suffix

    def load_dataset(self,
                     num_proc: int = 1, global_cfg: Namespace = None) -> \
            Dataset:
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
            datasets = concatenate_datasets([ds for _, ds in datasets.items()])
        ds = unify_format(datasets,
                          text_keys_to_load=self.text_keys_to_load,
                          text_key_to_process=global_cfg.text_key_to_process
                          if global_cfg else None,
                          num_proc=num_proc,
                          global_cfg=global_cfg)
        return ds


class RemoteFormatter(BaseFormatter):
    """The class is used to load a dataset from repository of huggingface
    hub."""

    def __init__(self,
                 dataset_path: str,
                 text_keys_to_load: List[str] = None,
                 **kwargs):
        """
        Initialization method.

        :param dataset_path: a dataset file or a dataset directory
        :param text_keys_to_load: key names of field that stores sample
            text.
        :param kwargs: extra args
        """
        self.path = dataset_path
        self.text_keys_to_load = text_keys_to_load
        self.kwargs = kwargs

    def load_dataset(self,
                     num_proc: int = 1,
                     global_cfg: Namespace = None) -> Dataset:
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
        ds = unify_format(ds,
                          text_keys_to_load=self.text_keys_to_load,
                          text_key_to_process=global_cfg.text_key_to_process,
                          num_proc=num_proc,
                          global_cfg=global_cfg)
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
    return datasets


def rename_ops_args_text_key(cfg, original_text_key, target_text_key):
    """
    Rename some ops args in cfg from a source text key to target text key.

    :param cfg: the global cfg used in consequent processes,
    :param original_text_key: source text key in op args :param modified
        cfg object
    """
    if not hasattr(cfg, 'process'):
        return
    process_list = cfg.process
    for i in range(len(process_list)):
        for op_name in process_list[i]:
            if process_list[i][op_name] and \
                    process_list[i][op_name]['text_key'] == original_text_key:
                process_list[i][op_name]['text_key'] = target_text_key


def unify_format(
    dataset: Dataset,
    text_keys_to_load: List[str] = 'deprecated',
    text_key_to_process: str = 'text',
    num_proc: int = 1,
    global_cfg: Namespace = None,
) -> Dataset:
    """
    Get an unified internal format, conduct the following modifications.

    1. based on the given keys, checking the key name of sample

    2. filter out those samples with empty or None text

    As a result, the dataset will being with the unified format such as:

    >>> {
    >>>     'text': 'hello-world',
    >>>     'text.instruction': "Let's think step by step.",
    >>>     "meta": {"date": 2012}
    >>>     'meta.src": "customized",
    >>>     "meta.version": "0.1",
    >>>     Fields.stats: {
    >>>         "lang": "en",
    >>>         "lang_score": 0.965
    >>>     }
    >>> }

    :param dataset: input dataset
    :param text_keys_to_load: original text key(s) of dataset, `deprecated`
    :param text_key_to_process: key name of field where the sample
        text to be processed,
    :param num_proc: number of processes for mapping
    :param global_cfg: the global cfg used in consequent processes,
        since cfg.text_key_to_process may need to be modified after unifying

    :return: unified_format_dataset
    """
    if isinstance(dataset, DatasetDict):
        datasets = list(dataset.values())
        assert len(datasets) == 1, 'Please make sure the passed datasets ' \
                                   'contains only 1 dataset'
        dataset = datasets[0]
    assert isinstance(dataset, Dataset), 'Currently we only support ' \
                                         'processing data with ' \
                                         "'huggingface-Dataset format'"

    if text_keys_to_load != 'deprecated':
        logger.warning('`text_keys_to_load` was deprecated, '
            'you can use text_key_to_process instead.')
        text_key_to_process = text_keys_to_load

    if text_key_to_process is None:
        text_key_to_process = ['text']

    if isinstance(text_key_to_process, str):
        text_key_to_process = [text_key_to_process]

    logger.info('Unifying the input dataset formats...')

    from data_juicer.core.data import NestedDataset
    dataset = NestedDataset(dataset)

    # 1. check text related keys
    for key in text_key_to_process:
        if key not in dataset.features:
            err_msg = f'There is no key [{key}] in dataset. You might set ' \
                      f'wrong text_key in the config file for your dataset. ' \
                      f'Please check and retry!'
            logger.error(err_msg)
            raise ValueError(err_msg)

    # update cfg
    if global_cfg:
        global_cfg.text_key_to_process = text_key_to_process[0]

    # 2. filter out those samples with empty or None text
    # TODO: optimize the filtering operation for better efficiency
    logger.info(f'There are {len(dataset)} sample(s) in the original dataset.')

    def non_empty_text(sample, target_keys):
        for target_key in target_keys:
            # TODO: case for SFT, in which the len(sample[target_key]) == 0
            if sample[target_key] is None:
                # we filter out the samples contains at least None column
                # since the op can not handle it now
                return False
        return True

    dataset = dataset.filter(
        non_empty_text,
        num_proc=num_proc,
        fn_kwargs={'target_keys': list(text_key_to_process)})
    logger.info(f'{len(dataset)} samples left after filtering empty text.')

    dataset.cleanup_cache_files()

    # 3. add Fields.stats field
    # TODO:
    # this is a temp solution,
    # it will occur errors when only call mapper ops
    # dataset = dataset.add_column( \
    # name=Fields.stats, column=[{}] * dataset.num_rows)

    return dataset


def load_formatter(dataset_path,
                   keys_to_load=None,
                   suffixes=None,
                   add_suffix=False,
                   **kwargs) -> BaseFormatter:
    """
    Load the appropriate formatter for different types of data formats.

    :param dataset_path: Path to dataset file or dataset directory
    :param keys_to_load: key names of field that stores sample text.
        Default: ['text']
    :param suffixes: the suffix of files that will be read. Default:
        None
    :return: a dataset formatter.
    """
    if keys_to_load is None:
        keys_to_load = ['text']
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
                                             text_keys_to_load=keys_to_load,
                                             suffixes=target_suffixes,
                                             add_suffix=add_suffix,
                                             **kwargs)

    # try huggingface dataset hub
    elif not is_absolute_path(dataset_path) and dataset_path.count('/') <= 1:
        return RemoteFormatter(dataset_path,
                               text_keys_to_load=keys_to_load,
                               **kwargs)

    # no data
    else:
        raise NotImplementedError
