import unittest
from loguru import logger
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.generate_qa_from_text_mapper import GenerateQAFromTextMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class GenerateQAFromTextMapperTest(DataJuicerTestCaseBase):
    text_key = 'text'

    def _run_op(self, enable_vllm=False, llm_params=None, sampling_params=None):
        op = GenerateQAFromTextMapper(
            enable_vllm=enable_vllm,
            llm_params=llm_params,
            sampling_params=sampling_params)
        
        samples = [{
            self.text_key: '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n'
        }]

        dataset = Dataset.from_list(samples)
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
        llm_params = {
            "tensor_parallel_size": torch.cuda.device_count(),
            "max_model_len": 1024,
            "max_num_seqs": 16
        }
        sampling_params={'temperature': 0.9, 'top_p': 0.95, 'max_tokens': 200}
        self._run_op(enable_vllm=True, llm_params=llm_params,sampling_params=sampling_params)


if __name__ == '__main__':
    unittest.main()
