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

    @classmethod
    def generate_dataset(cls, data, type='hf'):
        """Generate dataset for a specific executor.

        Args:
            type (str, optional): `hf` or `ray`. Defaults to "hf".
        """
        if type == 'hf':
            return Dataset.from_list(data)
        elif type == 'ray':
            return rd.from_items(data)

    @classmethod
    def run_single_op(cls, dataset, op, type='hf'):
        """Run operator in the specific executor."""
        if type == 'hf':
            if isinstance(op, Filter) and Fields.stats not in dataset.features:
                # TODO:
                # this is a temp solution,
                # only add stats when calling filter op
                dataset = dataset.add_column(name=Fields.stats,
                                             column=[{}] * dataset.num_rows)
            dataset = dataset.map(op.compute_stats)
            dataset = dataset.filter(op.process)
            dataset = dataset.select_columns(column_names=['text'])
            return dataset.to_list()
        elif type == 'ray':
            pass
