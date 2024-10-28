import unittest
from data_juicer.ops.mapper.optimize_instruction_mapper import OptimizeInstructionMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

class OptimizeInstructionMapperTest(DataJuicerTestCaseBase):

    text_key = 'text'

    def _run_optimize_instruction(self, enable_vllm=False):
        op = OptimizeInstructionMapper(
            hf_model='alibaba-pai/Qwen2-7B-Instruct-Refine',
            enable_vllm=enable_vllm
        )

        samples = [
            {self.text_key: '鱼香肉丝怎么做？'}
        ]

        for sample in samples:
            result = op.process(sample)
            print(f'Output results: {result}')
            self.assertIn(self.text_key, result)
        
    def test_optimize_instruction(self):
        self._run_optimize_instruction()

    def test_optimize_instruction_vllm(self):
        self._run_optimize_instruction(enable_vllm=True)


if __name__ == '__main__':
    unittest.main()
