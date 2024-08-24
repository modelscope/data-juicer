import unittest
from data_juicer.ops.mapper.sentence_augmentation_mapper import SentenceAugmentationMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class SentenceAugmentationMapperTest(DataJuicerTestCaseBase):

    text_key = 'text'

    def _run_sentence_augmentation_mapper(self):
        op = SentenceAugmentationMapper(
            hf_model='Qwen2-7B-Instruct',
            task_sentence="Please replace one entity in this sentence with another entity, such as an animal, a vehicle, or a piece of furniture. Please only answer with the replaced sentence. ASSISTANT:",
            max_new_tokens=512,
            temperature=0.9,
            top_p=0.95,
            num_beams=1,
        )

        samples = [
            {self.text_key: 'a book is near a cat and a dog'}
        ]

        for sample in samples:
            result = op.process(sample)
            print(f'Output results: {result}')
            self.assertIn(self.text_key, result)

    def test_sentence_augmentation_mapper(self):
        self._run_sentence_augmentation_mapper()



if __name__ == '__main__':
    unittest.main()