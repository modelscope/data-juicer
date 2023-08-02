import unittest

from data_juicer.ops.mapper.remove_specific_chars_mapper import \
    RemoveSpecificCharsMapper


class RemoveSpecificCharsMapperTest(unittest.TestCase):

    def setUp(self):
        self.op = RemoveSpecificCharsMapper()

    def _run_helper(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_complete_html_text(self):

        samples = [
            {
                'text': '这是一个干净的文本。Including Chinese and English.',
                'target': '这是一个干净的文本。Including Chinese and English.',
            },
            {
                'text': '◆●■►▼▲▴∆▻▷❖♡□',
                'target': '',
            },
            {
                'text': '►This is a dirty text ▻ 包括中文和英文',
                'target': 'This is a dirty text  包括中文和英文',
            },
            {
                'text': '多个●■►▼这样的特殊字符可以►▼▲▴∆吗？',
                'target': '多个这样的特殊字符可以吗？',
            },
            {
                'text': '未指定的●■☛₨➩►▼▲特殊字符会☻▷❖被删掉吗？？',
                'target': '未指定的☛₨➩特殊字符会☻被删掉吗？？',
            },
        ]
        self._run_helper(samples)


if __name__ == '__main__':
    unittest.main()
