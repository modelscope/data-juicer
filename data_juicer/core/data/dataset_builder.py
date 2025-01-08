import os
import shlex
from typing import List, Tuple, Union

import numpy as np
from datasets import concatenate_datasets

from data_juicer.core.data import NestedDataset
from data_juicer.core.data.config_validator import ConfigValidationError
from data_juicer.core.data.data_validator import DataValidatorRegistry
from data_juicer.core.data.load_strategy import DataLoadStrategyRegistry
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.utils.file_utils import is_absolute_path
from data_juicer.utils.sample import random_sample


class DatasetBuilder(object):
    """
    DatasetBuilder is a class that builds a dataset from a configuration.
    """

    def __init__(self, cfg, executor_type):
        self.cfg = cfg
        self.executor_type = executor_type

        # defaults to use dataset_path
        if cfg.dataset_path is not None:
            ds_configs = rewrite_cli_datapath(cfg.dataset_path)
        elif cfg.dataset is not None:
            ds_configs = cfg.dataset
        else:
            raise ConfigValidationError(
                'Unable to initialize dataset; should have one of '
                'dataset_path or dataset in configurations')

        # validate dataset config for type constraints
        # TODO other constraints; ray dataset only supports ondisk, etc.
        if type(ds_configs) != dict:
            raise ConfigValidationError(
                'Dataset config should be a dictionary')
        if 'configs' not in ds_configs:
            raise ConfigValidationError(
                'Dataset config should have a "configs" key')
        if (not isinstance(ds_configs['configs'], list)
                or len(ds_configs['configs']) == 0):
            raise ConfigValidationError(
                'Dataset config "configs" should be a non-empty list')
        if ('max_sample_num' in ds_configs
                and (type(ds_configs['max_sample_num']) != int
                     or ds_configs['max_sample_num'] <= 0)):
            raise ConfigValidationError(
                'Dataset config "max_sample_num" should be a positive integer')
        for ds_config in ds_configs['configs']:
            if type(ds_config) != dict:
                raise ConfigValidationError(
                    'Dataset configs should be dictionaries')
        types = [
            ds_config.get('type', None) for ds_config in ds_configs['configs']
        ]
        if len(set(types)) > 1:
            raise ConfigValidationError(
                'Mixture of diff types (ONDISK/REMOTE/...) are not supported')
        if types[0] == 'remote' and len(ds_configs['configs']) > 1:
            raise ConfigValidationError(
                'Multiple remote datasets are not supported')

        # initialize the data load strategies
        self.load_strategies = []
        for ds_config in ds_configs['configs']:
            # initialize data loading strategy
            data_type = ds_config.get('type', None)
            data_source = ds_config.get('source', None)
            self.load_strategies.append(
                DataLoadStrategyRegistry.get_strategy_class(
                    self.executor_type, data_type, data_source)(ds_config,
                                                                cfg=self.cfg))

        # initialzie the sample numbers
        self.max_sample_num = ds_configs.get('max_sample_num', None)
        # get weights and sample numbers
        if self.max_sample_num is not None:
            self.weights = [stra.weight for stra in self.load_strategies]
            self.sample_numbers = get_sample_numbers(self.weights,
                                                     self.max_sample_num)

        # initialize data validators
        self.validators = []
        if hasattr(cfg, 'validators'):
            for validator_config in cfg.validators:
                validator_type = validator_config['type']
                validator_cls = DataValidatorRegistry.get_validator(
                    validator_type)
                if validator_cls:
                    self.validators.append(validator_cls(validator_config))

    def load_dataset(self, **kwargs) -> Union[NestedDataset, RayDataset]:
        _datasets = []

        # load datasets with sample numbers
        for stra, weight, sample_num in zip(self.load_strategies, self.weights,
                                            self.sample_numbers):
            # load dataset with its load strategy
            dataset = stra.load_data(**kwargs)

            # do data validation
            for validator in self.validators:
                validator.validate(dataset)

            # do data sampling, if necessary
            if self.max_sample_num is not None:
                dataset = random_sample(dataset, weight, sample_num)

            _datasets.append(dataset)

        # handle data mixture
        if self.executor_type == 'local':
            return NestedDataset(concatenate_datasets(_datasets))
        elif self.executor_type == 'ray':
            return RayDataset(_datasets[0], )

    @classmethod
    def load_dataset_by_generated_config(cls, generated_dataset_config):
        """
        load dataset by generated config
        """
        assert isinstance(generated_dataset_config,
                          dict) and 'type' in generated_dataset_config
        args = generated_dataset_config.copy()

        # TODO finish the auto local dataset part
        obj_name = args.pop('type')
        from data_juicer.format.formatter import FORMATTERS
        dataset = FORMATTERS.modules[obj_name](**args).load_dataset()
        return dataset


def rewrite_cli_datapath(dataset_path, max_sample_num=None) -> List:
    """
    rewrite the dataset_path from CLI into proper dataset config format
    that is compatible with YAML config style; retrofitting CLI input
    of local files and huggingface path

    :param dataset_path: a dataset file or a dataset dir or a list of
        them, e.g. `<w1> ds1.jsonl <w2> ds2_dir <w3> ds3_file.json`
    :param max_sample_num: the maximum number of samples to load
    :return: list of dataset configs
    """
    paths, weights = parse_cli_datapath(dataset_path)
    ret = ({
        'configs': [],
        'max_sample_num': max_sample_num
    } if max_sample_num else {
        'configs': []
    })
    for p, w in zip(paths, weights):
        if os.path.isdir(p) or os.path.isfile(p):
            # local files
            ret['configs'].append({'type': 'ondisk', 'path': [p], 'weight': w})
        elif (not is_absolute_path(p) and not p.startswith('.')
              and p.count('/') <= 1):
            # remote huggingface
            ret['configs'].append({
                'type': 'huggingface',
                'path': p,
                'split': 'train'
            })
        else:
            #
            raise ValueError(
                f'Unable to load the dataset from [{dataset_path}]. '
                f'Data-Juicer CLI mode only supports local files '
                f'w or w/o weights, or huggingface path')
    return ret


def parse_cli_datapath(dataset_path) -> Tuple[List[str], List[float]]:
    """
    Split every dataset path and its weight.

    :param dataset_path: a dataset file or a dataset dir or a list of
        them, e.g. `<w1> ds1.jsonl <w2> ds2_dir <w3> ds3_file.json`
    :return: list of dataset path and list of weights
    """
    # Handle empty input
    if not dataset_path or not dataset_path.strip():
        return [], []

    # Use shlex to properly handle quoted strings
    try:
        tokens = shlex.split(dataset_path)
    except ValueError as e:
        raise ValueError(f'Invalid dataset path format: {e}')

    prefixes = []
    weights = []

    for i in range(len(tokens)):
        try:
            value = max(float(tokens[i]), 0.0)
            weights.append(value)
        except:  # noqa: E722
            value = tokens[i].strip()
            # if not set weight, use 1.0 as default
            if i == 0 or len(weights) == len(prefixes):
                weights.append(1.0)
            prefixes.append(value)

    return prefixes, weights


def get_sample_numbers(weights, max_sample_num):
    sample_numbers = [0] * len(weights)

    # Normalize weights
    weights = np.array(weights, dtype=np.float64)
    sum_weights = np.sum(weights)
    assert sum_weights > 0.0
    weights /= sum_weights
    sample_num_per_dataset = [
        int(np.ceil(max_sample_num * weight)) for weight in weights
    ]

    # Adjust
    acc_sample_numbers = 0
    for i in range(len(sample_num_per_dataset)):
        sample_numbers[i] = min(sample_num_per_dataset[i],
                                max_sample_num - acc_sample_numbers)
        acc_sample_numbers += sample_numbers[i]

    return sample_numbers
