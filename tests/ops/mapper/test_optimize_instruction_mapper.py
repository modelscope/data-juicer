import unittest
from loguru import logger
from data_juicer.ops.mapper.optimize_instruction_mapper import OptimizeInstructionMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class OptimizeInstructionMapperTest(DataJuicerTestCaseBase):
    query_key = 'query'

    def _run_optimize_instruction(self, enable_vllm=False):
        op = OptimizeInstructionMapper(
            hf_model='alibaba-pai/Qwen2-7B-Instruct-Refine',
            enable_vllm=enable_vllm
        )

        samples = [
            {self.query_key: '鱼香肉丝怎么做？'}
        ]

        for sample in samples:
            result = op.process(sample)
            logger.info(f'Output results: {result}')
            self.assertIn(self.query_key, result)
        
    def test_optimize_instruction(self):
        self._run_optimize_instruction()

    def test_optimize_instruction_vllm(self):
        self._run_optimize_instruction(enable_vllm=True)


if __name__ == '__main__':
    unittest.main()
