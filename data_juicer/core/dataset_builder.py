import os
from typing import List, Tuple, Union

from data_juicer.core.data import NestedDataset
from data_juicer.core.ray_data import RayDataset
from data_juicer.utils.file_utils import is_absolute_path


class DatasetBuilder(object):

    def __init__(self,
                 dataset_cfg,
                 max_samples=0,
                 generated_dataset_config=None,
                 text_keys=None,
                 suffixes=None,
                 add_suffix=False,
                 **kwargs):
        self.loaders = []
        # mixture or single

    def load_dataset(self) -> Union[NestedDataset, RayDataset]:
        # handle mixture dataset, nested dataset
        # handle sampling of mixture datasets
        #
        for f in self.formatters:
            f.load_dataset()
        return None


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
            ret.append({'type': 'local', 'path': [p], 'weight': w})
        elif not is_absolute_path(p) and not p.startswith(
                '.') and p.count('/') <= 1:
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
    data_prefix = dataset_path.split()
    prefixes = []
    weights = []

    for i in range(len(data_prefix)):
        try:
            value = max(float(data_prefix[i]), 0.0)
            weights.append(value)
        except:  # noqa: E722
            value = data_prefix[i].strip()
            # if not set weight, use 1.0 as default
            if i == 0 or len(weights) == len(prefixes):
                weights.append(1.0)
            prefixes.append(value)

    return prefixes, weights
