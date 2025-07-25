import os
import unittest

from loguru import logger

from data_juicer.ops.mapper.generate_prompt_from_examples_mapper import \
    GeneratePromptFromExamplesMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class GeneratePromptFromExamplesMapperTest(DataJuicerTestCaseBase):
    prompt_key = 'prompt'
    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')

    def _run_op(self, model="Qwen/Qwen2.5-7B-Instruct", enable_vllm=False, is_hf_model=True, sampling_params=None, num_proc=1):
        op = GeneratePromptFromExamplesMapper(
            api_or_hf_model=model,
            seed_file=os.path.join(self.root_path, 'demos/data/demo-dataset-prompts.jsonl'),
            example_num=3,
            example_score_key="score",
            enable_vllm=enable_vllm,
            is_hf_model=is_hf_model,
            sampling_params=sampling_params,
        )

        from data_juicer.format.empty_formatter import EmptyFormatter
        dataset = EmptyFormatter(3, [self.prompt_key]).load_dataset()

        results = dataset.map(op.process, num_proc=num_proc, with_rank=True)

        for row in results:
            logger.info(row)
            self.assertIn(op.prompt_key, row)

    def test(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(sampling_params=sampling_params)

    def test_api_model(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(model="qwen2.5-72b-instruct", is_hf_model=False, sampling_params=sampling_params)


if __name__ == '__main__':
    unittest.main()
