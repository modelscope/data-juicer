import unittest

from datasets import Dataset

from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class AlphanumericFilterTest(DataJuicerTestCaseBase):

    @TEST_TAG("single")
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

    @TEST_TAG("single")
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
