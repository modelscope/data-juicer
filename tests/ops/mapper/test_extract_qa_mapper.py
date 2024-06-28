import unittest
import json
from data_juicer.ops.mapper.extract_qa_mapper import ExtractQAMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class ExtractQAMapperTest(DataJuicerTestCaseBase):
    text_key = 'text'

    def _run_extract_qa(self, samples):
        op = ExtractQAMapper(
            hf_model='alibaba-pai/pai-qwen1_5-7b-doc2qa',
            qa_format='chatml'
            )
        for sample in samples:
            result = op.process(sample)
            out_text = json.loads(result[self.text_key])

            # test one output qa sample
            qa_sample = out_text[0]
            self.assertIn('role', qa_sample['messages'][0])
            self.assertIn('content', qa_sample['messages'][0])

    def test_extract_qa(self):
        samples = [
            {
            self.text_key: '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n'
            }]
        self._run_extract_qa(samples)


if __name__ == '__main__':
    unittest.main()
