import unittest

from datasets import Dataset

from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AlphanumericFilterTest(DataJuicerTestCaseBase):

    def _run_alphanumeric_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'a v s e e f g a qkc'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        tgt_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'a v s e e f g a qkc'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        dataset = DataJuicerTestCaseBase.generate_dataset(ds_list)
        op = AlphanumericFilter(min_ratio=0.2, max_ratio=0.9)
        result = DataJuicerTestCaseBase.run_single_op(dataset, op)
        self.assertEqual(result, tgt_list)

    def test_token_case(self):

        ds_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'a v s e e f g a qkc'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        tgt_list = [{
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'Do you need a cup of coffee?'
        }]
        dataset = DataJuicerTestCaseBase.generate_dataset(ds_list)
        op = AlphanumericFilter(tokenization=True, min_ratio=1.5)
        result = DataJuicerTestCaseBase.run_single_op(dataset, op)
        self.assertEqual(result, tgt_list)


if __name__ == '__main__':
    unittest.main()
