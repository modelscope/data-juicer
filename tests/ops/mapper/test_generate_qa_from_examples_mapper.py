import unittest

from loguru import logger

from data_juicer.ops.mapper.generate_qa_from_examples_mapper import \
    GenerateQAFromExamplesMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@unittest.skip('unknown vllm connection error')
class GenerateQAFromExamplesMapperTest(DataJuicerTestCaseBase):
    text_key = 'text'

    def _run_op(self, enable_vllm=False, sampling_params=None, num_proc=1):
        op = GenerateQAFromExamplesMapper(
            seed_file='demos/data/demo-dataset-chatml.jsonl',
            example_num=3,
            enable_vllm=enable_vllm,
            sampling_params=sampling_params,
        )

        from data_juicer.format.empty_formatter import EmptyFormatter
        dataset = EmptyFormatter(3, [self.text_key]).load_dataset()

        results = dataset.map(op.process, num_proc=num_proc, with_rank=True)

        for row in results:
            logger.info(row)
            self.assertIn(op.query_key, row)
            self.assertIn(op.response_key, row)

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
