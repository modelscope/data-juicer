import unittest

from data_juicer.ops.mapper.clean_email_mapper import CleanEmailMapper


class CleanEmailMapperTest(unittest.TestCase):

    def _run_clean_email(self, op, samples):
        for sample in samples:
            result = op.process(sample)
            self.assertEqual(result['text'], result['target'])

    def test_clean_email(self):

        samples = [{
            'text': 'happy day euqdh@cjqi.com',
            'target': 'happy day '
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁dasoidhao@1264fg.45om'
        }, {
            'text': 'ftp://examplema-nièrdash@hqbchd.ckdhnfes.cds',
            'target': 'ftp://examplema-niè'
        }, {
            'text': '👊23da44sh12@46hqb12chd.ckdhnfes.comd.dasd.asd.dc',
            'target': '👊'
        }]
        op = CleanEmailMapper()
        self._run_clean_email(op, samples)

    def test_replace_email(self):

        samples = [{
            'text': 'happy day euqdh@cjqi.com',
            'target': 'happy day <EMAIL>'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁dasoidhao@1264fg.45om'
        }, {
            'text': 'ftp://examplema-nièrdash@hqbchd.ckdhnfes.cds',
            'target': 'ftp://examplema-niè<EMAIL>'
        }, {
            'text': '👊23da44sh12@46hqb12chd.ckdhnfes.comd.dasd.asd.dc',
            'target': '👊<EMAIL>'
        }]
        op = CleanEmailMapper(repl='<EMAIL>')
        self._run_clean_email(op, samples)
        
if __name__ == '__main__':
    unittest.main()
