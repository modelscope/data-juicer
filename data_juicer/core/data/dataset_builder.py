import os
import shlex
from typing import List, Tuple, Union

from data_juicer.core.data import NestedDataset
from data_juicer.core.data.config_validator import ConfigValidationError
from data_juicer.core.data.data_validator import DataValidatorRegistry
from data_juicer.core.data.load_strategy import DataLoadStrategyRegistry
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.utils.file_utils import is_absolute_path


class DatasetBuilder(object):

    def __init__(self, cfg, executor_type):
        self.cfg = cfg
        self.executor_type = executor_type

        # defaults to use dataset_path
        if cfg.dataset_path is not None:
            ds_configs = rewrite_cli_datapath(cfg.dataset_path)
        elif cfg.dataset not in (None, []):
            ds_configs = cfg.dataset
        else:
            raise ConfigValidationError(
                'Unable to initialize dataset; should have one of '
                'dataset_path or dataset in configurations')

        # dataset config could be a list or a single entry; retrofit
        if not isinstance(ds_configs, list):
            ds_configs = [ds_configs]

        # validate dataset config for type constraints
        # 1. ds_config should be a dictionary
        # 2. ds_configs should only have one type
        # 3. if type is REMOTE, there should only one ds_config
        # TODO other constraints; ray dataset only supports ondisk, etc.
        for ds_config in ds_configs:
            if type(ds_config) != dict:
                raise ConfigValidationError(
                    'Dataset config should be a dictionary')
        types = [ds_config.get('type', None) for ds_config in ds_configs]
        if len(set(types)) > 1:
            raise ConfigValidationError(
                'Mixture of diff types (ONDISK/REMOTE/...) are not supported')
        if types[0] == 'remote' and len(ds_configs) > 1:
            raise ConfigValidationError(
                'Multiple remote datasets are not supported')

        # initialize the data load strategies
        self.load_strategies = []
        for ds_config in ds_configs:
            # initialize data loading strategy
            data_type = ds_config.get('type', None)
            data_source = ds_config.get('source', None)
            self.load_strategies.append(
                DataLoadStrategyRegistry.get_strategy_class(
                    self.executor_type, data_type, data_source)(ds_config))

        # initialize data validators
        self.validators = []
        if hasattr(cfg, 'validators'):
            for validator_config in cfg.validators:
                validator_type = validator_config['type']
                validator_cls = DataValidatorRegistry.get_validator(
                    validator_type)
                if validator_cls:
                    self.validators.append(validator_cls(validator_config))

    def load_dataset(self) -> Union[NestedDataset, RayDataset]:
        _datasets = []

        for f in self.load_strategies:
            # load dataset with its load strategy
            _dataset = f.load_data(self.cfg)

            # do data validation
            for validator in self.validators:
                validator.validate(_dataset)
            _datasets.append(_dataset)

        # handle data mixture; only supports ONDISK

        return _datasets[0]

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


def rewrite_cli_datapath(dataset_path) -> List:
    """
    rewrite the dataset_path from CLI into proper dataset config format
    that is compatible with YAML config style; retrofitting CLI input
    of local files and huggingface path

    :param dataset_path: a dataset file or a dataset dir or a list of
        them, e.g. `<w1> ds1.jsonl <w2> ds2_dir <w3> ds3_file.json`
    :return: list of dataset configs
    """
    paths, weights = parse_cli_datapath(dataset_path)
    ret = []
    for p, w in zip(paths, weights):
        if os.path.isdir(p) or os.path.isfile(p):
            # local files
            ret.append({'type': 'ondisk', 'path': [p], 'weight': w})
        elif (not is_absolute_path(p) and not p.startswith('.')
              and p.count('/') <= 1):
            # remote huggingface
            ret.append({'type': 'huggingface', 'path': p, 'split': 'train'})
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
