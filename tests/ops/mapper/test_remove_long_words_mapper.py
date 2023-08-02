import unittest

from data_juicer.ops.mapper.remove_long_words_mapper import \
    RemoveLongWordsMapper


class RemoveLongWordsMapperTest(unittest.TestCase):

    def _run_remove_long_words(self, samples, op):
        for sample in samples:
            result = op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_normal_case(self):

        samples = [{
            'text':
            'This paper proposed novel method LLM pretraining.',
            'target':
            'This paper proposed novel method LLM pretraining.'
        }]
        op = RemoveLongWordsMapper(min_len=3, max_len=15)
        self._run_remove_long_words(samples, op)

    def test_long_short_words_case(self):

        samples = [{
            'text':
            'This paper a novel eqeqweqwewqeqwe121e1 method on LLM pretrain.',
            'target': 'This paper novel method LLM pretrain.'
        }, {
            'text':
            'Sur la plateforme MT4, manières à ces fonctionnalités sont conçu',
            'target':
            'Sur plateforme MT4, manières ces fonctionnalités sont conçu'
        }]
        op = RemoveLongWordsMapper(min_len=3, max_len=15)
        self._run_remove_long_words(samples, op)

    def test_special_words_case(self):

        samples = [{
            'text':
            'This paper proposed a novel eqeqweqwewqenhq😊😠 method on LLM.',
            'target':
            'This paper proposed novel eqeqweqwewqenhq😊😠 method LLM.'
        }, {
            'text':
            "Sur la plateforme MT4, plusieurs manières d'accéder0123813976125",
            'target':
            "Sur plateforme MT4, plusieurs manières d'accéder0123813976125"
        }, {
            'text': 'The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.',
            'target': 'The Mona Lisa have eyebrows.'
        }]
        op = RemoveLongWordsMapper(min_len=3, max_len=15)
        self._run_remove_long_words(samples, op)


if __name__ == '__main__':
    unittest.main()
