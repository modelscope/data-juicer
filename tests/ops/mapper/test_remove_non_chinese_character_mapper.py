import unittest

from data_juicer.ops.mapper.remove_non_chinese_character_mapper import \
    RemoveNonChineseCharacterlMapper


class RemoveNonChineseCharacterlMapperrTest(unittest.TestCase):

    def setUp(self):
        self.op = RemoveNonChineseCharacterlMapper()

    def _run_remove_non_chinese_character(self, samples):
        for sample in samples:
            result = self.op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_remove_non_chinese_character(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁'
        }, {
            'text': 'ftp://exam匹配ple汉字ma-niè包括rdas繁體字h@hqbchd.ckdhnfes.cds',
            'target': '匹配汉字包括繁體字'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '所有的非汉字都会被去掉'
        }]
        self._run_remove_non_chinese_character(samples)


if __name__ == '__main__':
    unittest.main()
