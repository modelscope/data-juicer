from itertools import chain, repeat

import numpy as np
from datasets import Dataset
from loguru import logger

from data_juicer.ops.selector import (FrequencySpecifiedFieldSelector,
                                      TopkSpecifiedFieldSelector)


class SamplingMixin:

    def sample_data(self,
                    dataset_to_sample: Dataset = None,
                    load_data_np=None,
                    sample_ratio: float = 1.0,
                    sample_algo: str = 'uniform',
                    **kwargs):
        """
        Sample a subset from the given dataset.

        :param dataset_to_sample: Dataset to sample from. If None, will use
            the formatter linked by the executor. Default is None.
        :param load_data_np: number of workers when loading the dataset.
        :param sample_ratio: The ratio of the sample size to the original
            dataset size. Default is 1.0 (no sampling).
        :param sample_algo: Sampling algorithm to use. Options are "uniform",
            "frequency_specified_field_selector", or
            "topk_specified_field_selector".
            Default is "uniform".
        :return: A sampled Dataset.
        """
        # Determine the dataset to sample from
        if dataset_to_sample is not None:
            dataset = dataset_to_sample
        elif self.cfg.use_checkpoint and self.ckpt_manager.ckpt_available:
            logger.info('Loading dataset from checkpoint...')
            dataset = self.ckpt_manager.load_ckpt()
        elif hasattr(self, 'formatter'):
            logger.info('Loading dataset from data formatter...')
            if load_data_np is None:
                load_data_np = self.cfg.np
            dataset = self.formatter.load_dataset(load_data_np, self.cfg)
        else:
            raise ValueError('No dataset available to sample from.')

        # Perform sampling based on the specified algorithm
        if sample_algo == 'uniform':
            return random_sample(dataset, sample_ratio)
        elif sample_algo == 'frequency_specified_field_selector':
            dj_op = FrequencySpecifiedFieldSelector(**kwargs)
            return dj_op.process(dataset)
        elif sample_algo == 'topk_specified_field_selector':
            dj_op = TopkSpecifiedFieldSelector(**kwargs)
            return dj_op.process(dataset)
        else:
            raise ValueError(f'Unsupported sample_algo: {sample_algo}')


def random_sample(dataset, weight=1.0, sample_number=0, seed=None):
    """
    Randomly sample a subset from a dataset with weight or number,
    if sample number is bigger than 0, we will use sample
    number instead of weight.
    :param dataset: a HuggingFace dataset
    :param weight: sample ratio of dataset
    :param sample_number: sample number of dataset
    :param seed: random sample seed, if None, 42 as default
    :return: a subset of dataset
    """
    if seed is None:
        seed = 42

    ds_samples = dataset.num_rows
    if sample_number <= 0:
        sample_number = int(np.ceil(ds_samples * weight))

    if sample_number == ds_samples:
        return dataset

    sample_index = range(sample_number)

    n_repeat = int(np.ceil(sample_number / ds_samples)) - 1
    if n_repeat > 0:
        remain_samples = sample_number - n_repeat * ds_samples
        sample_index = chain(*repeat(range(ds_samples), n_repeat),
                             range(remain_samples))

    return dataset.shuffle(seed=seed).select(sample_index)
