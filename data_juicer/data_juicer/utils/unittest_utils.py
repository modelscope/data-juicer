import os
import shutil
import unittest

import numpy
import pyarrow as pa
import ray.data as rd
from datasets import Dataset

from data_juicer.ops import Filter
from data_juicer.utils.constant import Fields
from data_juicer.utils.registry import Registry

SKIPPED_TESTS = Registry('SkippedTests')


def TEST_TAG(*tags):
    """Tags for test case.
    Currently, `standalone`, `ray` are supported.
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

    def generate_dataset(self, data):
        """Generate dataset for a specific executor.

        Args:
            type (str, optional): "standalone" or "ray".
            Defaults to "standalone".
        """
        current_tag = getattr(self, 'current_tag', 'standalone')
        if current_tag.startswith('standalone'):
            return Dataset.from_list(data)
        elif current_tag.startswith('ray'):
            dataset = rd.from_items(data)
            if Fields.stats not in dataset.columns(fetch_if_missing=False):

                def process_batch_arrow(table: pa.Table) -> pa.Table:
                    new_column_data = [{} for _ in range(len(table))]
                    new_talbe = table.append_column(Fields.stats,
                                                    [new_column_data])
                    return new_talbe

                dataset = dataset.map_batches(process_batch_arrow,
                                              batch_format='pyarrow')
            return dataset
        else:
            raise ValueError('Unsupported type')

    def run_single_op(self, dataset, op, column_names):
        """Run operator in the specific executor."""
        current_tag = getattr(self, 'current_tag', 'standalone')
        if current_tag.startswith('standalone'):
            if isinstance(op, Filter) and Fields.stats not in dataset.features:
                dataset = dataset.add_column(name=Fields.stats,
                                             column=[{}] * dataset.num_rows)
            dataset = dataset.map(op.compute_stats)
            dataset = dataset.filter(op.process)
            dataset = dataset.select_columns(column_names=column_names)
            return dataset.to_list()
        elif current_tag.startswith('ray'):
            dataset = dataset.map(op.compute_stats)
            dataset = dataset.filter(op.process)
            dataset = dataset.to_pandas().get(column_names)
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
