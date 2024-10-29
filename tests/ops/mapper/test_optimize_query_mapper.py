import unittest

from loguru import logger

from data_juicer.ops.mapper.optimize_query_mapper import OptimizeQueryMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)


# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class OptimizeQueryMapperTest(DataJuicerTestCaseBase):

    def _run_op(self,
                enable_vllm=False,
                llm_params=None,
                sampling_params=None):
        op = OptimizeQueryMapper(
            hf_model='alibaba-pai/Qwen2-7B-Instruct-Refine',
            enable_vllm=enable_vllm,
            llm_params=llm_params,
            sampling_params=sampling_params)

        samples = [{
            'query':
            '鱼香肉丝怎么做？',
            'response':
            '鱼香肉丝是将猪肉丝与胡萝卜、青椒、木耳炒制，调入调味料如酱油、醋和辣豆瓣酱，快速翻炒而成的美味佳肴。'
        }]

        for sample in samples:
            result = op.process(sample)
            logger.info(f'Output results: {result}')
            self.assertNotEqual(result['query'], '')

    def test(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(sampling_params=sampling_params)

    def test_vllm(self):
        import torch
        llm_params = {'tensor_parallel_size': torch.cuda.device_count()}
        sampling_params = {'max_tokens': 200}
        self._run_op(enable_vllm=True,
                     llm_params=llm_params,
                     sampling_params=sampling_params)


if __name__ == '__main__':
    unittest.main()