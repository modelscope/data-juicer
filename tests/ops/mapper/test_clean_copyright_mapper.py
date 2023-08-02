import unittest

from data_juicer.ops.mapper.clean_copyright_mapper import CleanCopyrightMapper


class CleanCopyrightMapperTest(unittest.TestCase):

    def setUp(self):
        self.op = CleanCopyrightMapper()

    def _run_clean_copyright(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_clean_copyright(self):

        samples = [{
            'text': '这是一段 /* 多行注释\n注释内容copyright\n*/ 的文本。另外还有一些 // 单行注释。',
            'target': '这是一段  的文本。另外还有一些 // 单行注释。'
        }, {
            'text': '如果多行/*注释中没有\n关键词,那么\n这部分注释也不会\n被清除*/\n会保留下来',
            'target': '如果多行/*注释中没有\n关键词,那么\n这部分注释也不会\n被清除*/\n会保留下来'
        }, {
            'text': '//if start with\n//that will be cleand \n envenly',
            'target': ' envenly'
        }, {
            'text': 'http://www.nasosnsncc.com',
            'target': 'http://www.nasosnsncc.com'
        }, {
            'text': '#if start with\nthat will be cleand \n#envenly',
            'target': 'that will be cleand \n#envenly'
        }, {
            'text': '--if start with\n--that will be cleand \n#envenly',
            'target': ''
        }]
        self._run_clean_copyright(samples)


if __name__ == '__main__':
    unittest.main()
