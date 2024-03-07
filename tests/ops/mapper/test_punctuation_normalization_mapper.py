import unittest

from data_juicer.ops.mapper.punctuation_normalization_mapper import \
    PunctuationNormalizationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class PunctuationNormalizationMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        self.op = PunctuationNormalizationMapper()

    def _run_punctuation_normalization(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_case(self):

        samples = [{
            'text':
            '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►',
            'target':
            ",.,\"\"\"\"\"\"\"\"\"\"'::?!();- - . ~'...-<>[]%-"
        }]

        self._run_punctuation_normalization(samples)


if __name__ == '__main__':
    unittest.main()
