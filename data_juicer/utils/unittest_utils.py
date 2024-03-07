import os
import shutil
import unittest

from data_juicer.utils.registry import Registry

SKIPPED_TESTS = Registry('SkippedTests')


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
