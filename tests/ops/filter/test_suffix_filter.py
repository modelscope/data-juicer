import unittest

from datasets import Dataset

from data_juicer.ops.filter.suffix_filter import SuffixFilter


class SuffixFilterTest(unittest.TestCase):

    def _run_suffix_filter(self, dataset: Dataset, target_list, op):
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'text': 'Today is Sun',
            'meta': {
                'suffix': '.pdf'
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'meta': {
                'suffix': '.docx'
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'meta': {
                'suffix': '.txt'
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'meta': {
                'suffix': '.html'
            }
        }, {
            'text': 'dasdasdasdasdasdasdasd',
            'meta': {
                'suffix': '.py'
            }
        }]
        tgt_list = [{
            'text': 'Today is Sun',
            'meta': {
                'suffix': '.pdf'
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'meta': {
                'suffix': '.txt'
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = SuffixFilter(suffixes=['.txt', '.pdf'])
        self._run_suffix_filter(dataset, tgt_list, op)

    def test_none_case(self):

        ds_list = [{
            'text': 'Today is Sun',
            'meta': {
                'suffix': '.pdf'
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'meta': {
                'suffix': '.docx'
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'meta': {
                'suffix': '.txt'
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'meta': {
                'suffix': '.html'
            }
        }, {
            'text': 'dasdasdasdasdasdasdasd',
            'meta': {
                'suffix': '.py'
            }
        }]
        tgt_list = [{
            'text': 'Today is Sun',
            'meta': {
                'suffix': '.pdf'
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'meta': {
                'suffix': '.docx'
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'meta': {
                'suffix': '.txt'
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'meta': {
                'suffix': '.html'
            }
        }, {
            'text': 'dasdasdasdasdasdasdasd',
            'meta': {
                'suffix': '.py'
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = SuffixFilter()
        self._run_suffix_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
