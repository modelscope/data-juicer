import os
import shutil
import unittest


class DataJuicerTestCaseBase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls, hf_model_name=None) -> None:
        # clean the huggingface model cache files
        import transformers
        if hf_model_name:
            # given the hf model name, remove this model only
            print(f'CLEAN model cache files for {hf_model_name}')
            model_dir = os.path.join(
                transformers.TRANSFORMERS_CACHE,
                f'models--{hf_model_name.replace("/", "--")}'
            )
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
        else:
            # not giben the hf model name, remove the whole TRANSFORMERS_CACHE
            print('CLEAN all TRANSFORMERS_CACHE')
            if os.path.exists(transformers.TRANSFORMERS_CACHE):
                shutil.rmtree(transformers.TRANSFORMERS_CACHE)
