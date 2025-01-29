from typing import List, Union

import numpy as np
from datasets import Dataset, concatenate_datasets
from loguru import logger

from data_juicer.format.formatter import BaseFormatter
from data_juicer.utils.sample import random_sample


class MixtureFormatter(BaseFormatter):
    """The class mixes multiple datasets by randomly selecting samples from
    every dataset and merging them, and then exports the merged datasset as a
    new mixed dataset."""

    def __init__(self,
                 dataset_path: str,
                 suffixes: Union[str, List[str], None] = None,
                 text_keys=None,
                 add_suffix=False,
                 max_samples=None,
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
        :param max_samples: max samples number of mixed dataset.
        :param kwargs: extra args
        """

        data_prefixes, weights = self._get_weight(data_prefix=dataset_path)
        sample_numbers = [0] * len(weights)
        if max_samples is not None:
            # Normalize weights.
            weights = np.array(weights, dtype=np.float64)
            sum_weights = np.sum(weights)
            assert sum_weights > 0.0
            weights /= sum_weights
            sample_num_per_dataset = [
                int(np.ceil(max_samples * weight)) for weight in weights
            ]

            # Adjust
            acc_sample_numbers = 0
            for i in range(len(sample_num_per_dataset)):
                sample_numbers[i] = min(sample_num_per_dataset[i],
                                        max_samples - acc_sample_numbers)
                acc_sample_numbers += sample_numbers[i]

        self.sample_numbers = sample_numbers
        self.weights = weights
        self.formatters = None
        # [
        #     load_formatter(dataset_path=data_prefix,
        #                    suffixes=suffixes,
        #                    text_keys=text_keys,
        #                    add_suffix=add_suffix,
        #                    **kwargs) for data_prefix in data_prefixes
        # ]

    def load_dataset(self, num_proc: int = 1, global_cfg=None) -> Dataset:
        """
        Load a mixed dataset.

        :param num_proc: number of processes when loading the dataset
        :param global_cfg: the global cfg used in consequent processes,
        :return: mixed dataset
        """
        dataset_list = []
        for weight, sample_num, formatter in zip(self.weights,
                                                 self.sample_numbers,
                                                 self.formatters):
            dataset = formatter.load_dataset(num_proc, global_cfg)
            sampled = random_sample(dataset, weight, sample_num)
            logger.info(f'sampled {len(sampled)} from '
                        f'{len(dataset)}')
            dataset_list.append(sampled)

        from data_juicer.core.data import NestedDataset
        mixed_dataset = NestedDataset(concatenate_datasets(dataset_list))
        logger.info(f'There are {len(mixed_dataset)} in final dataset')
        return mixed_dataset
