import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.optimize_qa_mapper import OptimizeQAMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)


# Skip tests for this OP in the GitHub actions due to ?
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class OptimizeQAMapperTest(DataJuicerTestCaseBase):

    def _run_op(self, enable_vllm=False, sampling_params=None, num_proc=1):

        op = OptimizeQAMapper(enable_vllm=enable_vllm,
                              sampling_params=sampling_params)

        samples = [{
            'query':
            '鱼香肉丝怎么做？',
            'response':
            '鱼香肉丝是将猪肉丝与胡萝卜、青椒、木耳炒制，调入调味料如酱油、醋和辣豆瓣酱，快速翻炒而成的美味佳肴。'
        }, {
            'query': '什么是蚂蚁上树？',
            'response': '蚂蚁上树是一道中国菜。'
        }]
        dataset = Dataset.from_list(samples)
        results = dataset.map(op.process, num_proc=num_proc, with_rank=True)

        for row in results:
            logger.info(f'Output results: {row}')
            self.assertNotEqual(row['query'], '')
            self.assertNotEqual(row['response'], '')

    def test(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(sampling_params=sampling_params)

    def test_multi_process(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(sampling_params=sampling_params, num_proc=2)

    def test_vllm(self):
        sampling_params = {'max_tokens': 200}
        self._run_op(enable_vllm=True, sampling_params=sampling_params)


if __name__ == '__main__':
    unittest.main()
