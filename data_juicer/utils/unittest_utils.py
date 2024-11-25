import os
import shutil
import unittest

import numpy

from data_juicer import is_cuda_available
from data_juicer.core.data import DJDataset, NestedDataset
from data_juicer.core.ray_data import RayDataset
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import free_models
from data_juicer.utils.registry import Registry

rd = LazyLoader('rd', 'ray.data')
transformers = LazyLoader('transformers', 'transformers')

SKIPPED_TESTS = Registry('SkippedTests')

CLEAR_MODEL = False


def TEST_TAG(*tags):
    """Tags for test case.
    Currently, `standalone`, `ray` are supported.
    """

    def decorator(func):
        setattr(func, '__test_tags__', tags)
        return func

    return decorator


def set_clear_model_flag(flag):
    global CLEAR_MODEL
    CLEAR_MODEL = flag
    if CLEAR_MODEL:
        print('CLEAR DOWNLOADED MODELS AFTER UNITTESTS.')
    else:
        print('KEEP DOWNLOADED MODELS AFTER UNITTESTS.')


class DataJuicerTestCaseBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set maxDiff for all test cases based on an environment variable
        max_diff = os.getenv('TEST_MAX_DIFF', 'None')
        cls.maxDiff = None if max_diff == 'None' else int(max_diff)

        import multiprocess
        cls.original_mp_method = multiprocess.get_start_method()
        if is_cuda_available():
            multiprocess.set_start_method('spawn', force=True)

    @classmethod
    def tearDownClass(cls, hf_model_name=None) -> None:
        import multiprocess
        multiprocess.set_start_method(cls.original_mp_method, force=True)

        # clean the huggingface model cache files
        if not CLEAR_MODEL:
            pass
        elif hf_model_name:
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
    def tearDown(cls) -> None:
        free_models()

    def generate_dataset(self, data) -> DJDataset:
        """Generate dataset for a specific executor.

        Args:
            type (str, optional): "standalone" or "ray".
            Defaults to "standalone".
        """
        current_tag = getattr(self, 'current_tag', 'standalone')
        if current_tag.startswith('standalone'):
            return NestedDataset.from_list(data)
        elif current_tag.startswith('ray'):
            dataset = rd.from_items(data)
            return RayDataset(dataset)
        else:
            raise ValueError('Unsupported type')

    def run_single_op(self, dataset: DJDataset, op, column_names):
        """Run operator in the specific executor."""
        current_tag = getattr(self, 'current_tag', 'standalone')
        dataset = dataset.process(op)
        if current_tag.startswith('standalone'):
            dataset = dataset.select_columns(column_names=column_names)
            return dataset.to_list()
        elif current_tag.startswith('ray'):
            dataset = dataset.data.to_pandas().get(column_names)
            if dataset is None:
                return []
            return dataset.to_dict(orient='records')
        else:
            raise ValueError('Unsupported type')

    def assertDatasetEqual(self, first, second):

        def convert_record(rec):
            for key in rec.keys():
                # Convert incomparable `list` to comparable `tuple`
                if isinstance(rec[key], numpy.ndarray) or isinstance(
                        rec[key], list):
                    rec[key] = tuple(rec[key])
            return rec

        first = [convert_record(d) for d in first]
        second = [convert_record(d) for d in second]
        first = sorted(first, key=lambda x: tuple(sorted(x.items())))
        second = sorted(second, key=lambda x: tuple(sorted(x.items())))
        return self.assertEqual(first, second)
