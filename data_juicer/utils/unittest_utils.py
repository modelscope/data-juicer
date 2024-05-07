import os
import shutil
import unittest

import ray.data as rd
from datasets import Dataset

from data_juicer.ops import Filter
from data_juicer.utils.constant import Fields
from data_juicer.utils.registry import Registry

SKIPPED_TESTS = Registry('SkippedTests')


def TEST_TAG(*tags):
    """Tags for test case.
    Currently, `standalone`, `ray`, `standalone-gpu`, `ray-gpu` are supported.
    """

    def decorator(func):
        setattr(func, '__test_tags__', tags)
        return func

    return decorator


class DataJuicerTestCaseBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set maxDiff for all test cases based on an environment variable
        max_diff = os.getenv('TEST_MAX_DIFF', 'None')
        cls.maxDiff = None if max_diff == 'None' else int(max_diff)

    @classmethod
    def tearDownClass(cls, hf_model_name=None) -> None:
        # clean the huggingface model cache files
        import transformers
        if hf_model_name:
            # given the hf model name, remove this model only
            model_dir = os.path.join(
                transformers.TRANSFORMERS_CACHE,
                f'models--{hf_model_name.replace("/", "--")}')
            if os.path.exists(model_dir):
                print(f'CLEAN model cache files for {hf_model_name}')
                shutil.rmtree(model_dir)
        else:
            # not given the hf model name, remove the whole TRANSFORMERS_CACHE
            if os.path.exists(transformers.TRANSFORMERS_CACHE):
                print('CLEAN all TRANSFORMERS_CACHE')
                shutil.rmtree(transformers.TRANSFORMERS_CACHE)

    def generate_dataset(cls, data, type='standalone'):
        """Generate dataset for a specific executor.

        Args:
            type (str, optional): `hf` or `ray`. Defaults to "hf".
        """
        if type.startswith('standalone'):
            return Dataset.from_list(data)
        elif type.startswith('ray'):
            return rd.from_items(data)
        else:
            raise ValueError('Unsupported type')

    def run_single_op(cls, dataset, op, column_names, type='standalone'):
        """Run operator in the specific executor."""
        if type.startswith('standalone'):
            if isinstance(op, Filter) and Fields.stats not in dataset.features:
                dataset = dataset.add_column(name=Fields.stats,
                                             column=[{}] * dataset.num_rows)
            dataset = dataset.map(op.compute_stats)
            dataset = dataset.filter(op.process)
            dataset = dataset.select_columns(column_names=column_names)
            return dataset.to_list()
        elif type.startswith('ray'):
            raise ValueError('Unsupported type')
        else:
            raise ValueError('Unsupported type')
