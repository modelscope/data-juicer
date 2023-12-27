import unittest

from datasets import Dataset

from data_juicer.ops.filter.language_id_score_filter import \
    LanguageIDScoreFilter
from data_juicer.utils.constant import Fields


class LanguageIDScoreFilterTest(unittest.TestCase):

    def _run_language_id_score_filter(self, dataset: Dataset, target_list, op):
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

    def test_en_case(self):

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
        dataset = Dataset.from_list(ds_list)
        op = LanguageIDScoreFilter(lang='en', min_score=0.8)
        self._run_language_id_score_filter(dataset, tgt_list, op)

    def test_zh_case(self):

        ds_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': '我出生于2023年12月15日'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—'
        }, {
            'text': '他的英文名字叫Harry Potter'
        }, {
            'text': '这是一个测试'
        }]
        tgt_list = [{
            'text': '我出生于2023年12月15日'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—'
        }, {
            'text': '他的英文名字叫Harry Potter'
        }, {
            'text': '这是一个测试'
        }]
        dataset = Dataset.from_list(ds_list)
        op = LanguageIDScoreFilter(lang='zh', min_score=0.8)
        self._run_language_id_score_filter(dataset, tgt_list, op)

    def test_none_case(self):

        ds_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': '我出生于2023年12月15日'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—'
        }, {
            'text': '他的英文名字叫Harry Potter'
        }, {
            'text': '这是一个测试'
        }]
        tgt_list = [{
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': '我出生于2023年12月15日'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—'
        }, {
            'text': '他的英文名字叫Harry Potter'
        }, {
            'text': '这是一个测试'
        }]
        dataset = Dataset.from_list(ds_list)
        op = LanguageIDScoreFilter(lang='', min_score=0.8)
        self._run_language_id_score_filter(dataset, tgt_list, op)

    def test_multi_language_case(self):

        ds_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text': '我出生于2023年12月15日'
        }, {
            'text': '他的英文名字叫Harry Potter'
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
        }, {
            'text': '这是一个测试'
        }]
        tgt_list = [{
            'text': '我出生于2023年12月15日'
        }, {
            'text': '他的英文名字叫Harry Potter'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': '这是一个测试'
        }]
        dataset = Dataset.from_list(ds_list)
        op = LanguageIDScoreFilter(lang=['en', 'zh'], min_score=0.8)
        self._run_language_id_score_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
