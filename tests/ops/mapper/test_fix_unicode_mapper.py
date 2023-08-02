import unittest

from data_juicer.ops.mapper.fix_unicode_mapper import FixUnicodeMapper


class FixUnicodeMapperTest(unittest.TestCase):

    def setUp(self):
        self.op = FixUnicodeMapper()

    def _run_fix_unicode(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_bad_unicode_text(self):

        samples = [
            {
                'text': 'âœ” No problems',
                'target': '✔ No problems'
            },
            {
                'text':
                'The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.',
                'target': 'The Mona Lisa doesn\'t have eyebrows.'
            },
        ]

        self._run_fix_unicode(samples)

    def test_good_unicode_text(self):
        samples = [
            {
                'text': 'No problems',
                'target': 'No problems'
            },
            {
                'text': '阿里巴巴',
                'target': '阿里巴巴'
            },
        ]
        self._run_fix_unicode(samples)


if __name__ == '__main__':
    unittest.main()
