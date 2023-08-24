from typing import List, Tuple, Union

import numpy as np
from datasets import Dataset, concatenate_datasets
from loguru import logger

from .formatter import BaseFormatter, load_formatter


class MixtureFormatter(BaseFormatter):
    """The class mixes multiple datasets by randomly selecting samples from
    every dataset and merging them, and then exports the merged datasset as a
    new mixed dataset."""

    def __init__(self,
                 dataset_path: str,
                 suffixes: Union[str, List[str], Tuple[str]] = None,
                 text_keys=None,
                 add_suffix=False,
                 **kwargs):
        """
        Initialization method.

        :param dataset_path: a dataset file or a dataset dir or a list
            of them, optional weights, default 1.0 e.g. `<w1> ds.jsonl
            <w2> ds_dir <w3> ds_file.json`
        :param suffixes: files with specified suffixes to be processed
        :param text_keys: key names of field that stores sample text.
        :param add_suffix: whether to add the file suffix to dataset
            meta info
        :param kwargs: extra args
        """
        data_prefixes, weights = self._get_weight(data_prefix=dataset_path)
        self.weights = weights
        self.formatters = [
            load_formatter(dataset_path=data_prefix,
                           suffixes=suffixes,
                           text_keys=text_keys,
                           add_suffix=add_suffix,
                           **kwargs) for data_prefix in data_prefixes
        ]

    def _get_weight(self, data_prefix):
        """
        Split every dataset path and its weight.

        :param data_prefix: a dataset file or a dataset dir or a list of
            them, e.g. `<w1> ds1.jsonl <w2> ds2_dir <w3> ds3_file.json`
        :return: list of dataset path and list of weights
        """
        data_prefix = data_prefix.split()
        weights = []
        prefixes = []

        for i in range(len(data_prefix)):
            try:
                value = float(data_prefix[i])
                weights.append(value)
            except:  # noqa: E722
                value = data_prefix[i].strip()

                # if not set weight, use 1.0 as default
                if i == 0 or len(weights) == len(prefixes):
                    weights.append(1.0)
                prefixes.append(value)
        return prefixes, weights

    def _random_sample(self, dataset, weight=1.0, seed=None):
        """
        Randomly sample a subset from a dataset with weight.
        :param dataset: a HuggingFace dataset
        :param weight: sample ratio of dataset
        :param seed: random sample seed, if None, 42 as default
        :return: a subset of dataset
        """
        if seed is None:
            seed = 42
        num_samples = min(int(np.ceil(dataset.num_rows * weight)),
                          dataset.num_rows)
        if num_samples == dataset.num_rows:
            return dataset
        return dataset.shuffle(seed=seed).select(range(num_samples))

    def load_dataset(self, num_proc: int = 1) -> Dataset:
        """
        Load a mixed dataset.

        :param num_proc: number of processes when loading the dataset
        :return: mixed dataset
        """
        dataset_list = []
        for weight, formatter in zip(self.weights, self.formatters):
            dataset = formatter.load_dataset(num_proc)
            sampled = self._random_sample(dataset, weight)
            logger.info(f'sampled {len(sampled)} from '
                        f'{len(dataset)} with weight {weight}')
            dataset_list.append(sampled)

        from data_juicer.core.data import NestedDataset
        mixed_dataset = NestedDataset(concatenate_datasets(dataset_list))
        logger.info(f'There are {len(mixed_dataset)} in final dataset')
        return mixed_dataset
