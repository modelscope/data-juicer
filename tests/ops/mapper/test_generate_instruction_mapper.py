import unittest
import json
from data_juicer.ops.mapper.generate_instruction_mapper import GenerateInstructionMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

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

        for item in dataset:            
            out_sample = json.loads(item[self.text_key])
            print(f'Output sample: {out_sample}')
            # test one output qa sample
            self.assertIn('role', out_sample['messages'][0])
            self.assertIn('content', out_sample['messages'][0])
        
    def test_generate_instruction(self):
        self._run_generate_instruction()

    def test_generate_instruction_vllm(self):
        self._run_generate_instruction(enable_vllm=True)


if __name__ == '__main__':
    unittest.main()
