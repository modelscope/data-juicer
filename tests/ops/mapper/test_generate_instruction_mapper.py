import unittest
import json
from loguru import logger
from data_juicer.ops.mapper.generate_instruction_mapper import GenerateInstructionMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class GenerateInstructionMapperTest(DataJuicerTestCaseBase):

    text_key = 'text'

    def _run_generate_instruction(self, enable_vllm=False):
        op = GenerateInstructionMapper(
            hf_model='Qwen/Qwen-7B-Chat',
            seed_file='demos/data/demo-dataset-chatml.jsonl',
            instruct_num=2,
            trust_remote_code=True,
            enable_vllm=enable_vllm
        )

        from data_juicer.format.empty_formatter import EmptyFormatter
        dataset = EmptyFormatter(3, [self.text_key]).load_dataset()

        dataset = dataset.map(op.process)

        for row in dataset:
            logger.info(row)
            # Note: If switching models causes this assert to fail, it may not be a code issue; 
            # the model might just have limited capabilities.
            self.assertNotEqual(row[op.query_key], '')
            self.assertNotEqual(row[op.response_key], '')

    def test_generate_instruction(self):
        self._run_generate_instruction()

    def test_generate_instruction_vllm(self):
        self._run_generate_instruction(enable_vllm=True)


if __name__ == '__main__':
    unittest.main()
