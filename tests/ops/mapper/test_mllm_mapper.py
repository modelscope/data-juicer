import unittest
from data_juicer.ops.mapper.mllm_mapper import MllmMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class MllmMapperTest(DataJuicerTestCaseBase):

    text_key = 'text'
    image_key = "images"

    def _run_mllm(self, enable_vllm=False):
        op = MllmMapper(
            hf_model='llava-v1.6-vicuna-7b-hf',
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=512
        )

        samples = [
            {self.text_key: 'Describe this image.', self.image_key: ["./ipod.jpg", "./crayon.jpg"]},
        ]

        for sample in samples:
            result = op.process(sample)
            print(f'Output results: {result}')

    def test_mllm(self):
        self._run_mllm()


if __name__ == '__main__':
    unittest.main()