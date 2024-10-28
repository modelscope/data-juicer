import unittest
import json
from loguru import logger
from data_juicer.ops.mapper.generate_qa_from_examples_mapper import GenerateQAFromExamplesMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class GenerateQAFromExamplesMapperTest(DataJuicerTestCaseBase):
    text_key = 'text'

    def _run_op(self, enable_vllm=False, llm_params=None, sampling_params=None):
        op = GenerateQAFromExamplesMapper(
            seed_file='demos/data/demo-dataset-chatml.jsonl',
            example_num=3,
            enable_vllm=enable_vllm,
            llm_params=llm_params,
            sampling_params=sampling_params,
        )

        from data_juicer.format.empty_formatter import EmptyFormatter
        dataset = EmptyFormatter(3, [self.text_key]).load_dataset()

        dataset = dataset.map(op.process)

        for row in dataset:
            logger.info(row)
            # Note: If switching models causes this assert to fail, it may not be a code issue; 
            # the model might just have limited capabilities.
            self.assertIn(op.query_key, row)
            self.assertIn(op.response_key, row)

    def test(self):
        sampling_params = {"max_new_tokens": 200}
        self._run_op(sampling_params=sampling_params)

    def test_vllm(self):
        import torch
        llm_params = {"tensor_parallel_size": torch.cuda.device_count()}
        sampling_params = {"max_tokens": 200}
        self._run_op(enable_vllm=True, llm_params=llm_params, sampling_params=sampling_params)


if __name__ == '__main__':
    unittest.main()
