import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.text_action_filter import TextActionFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TextActionFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    cat_path = os.path.join(data_path, 'cat.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    def _run_text_action_filter(self, dataset: Dataset, target_list, op,
                                column_names):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=column_names)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_en_text_case(self):

        ds_list = [{
            'text': 'Tom is playing piano.'
        }, {
            'text': 'Tom plays piano.'
        }, {
            'text': 'Tom played piano.'
        }, {
            'text': 'I play piano.'
        }, {
            'text': 'to play piano.'
        }, {
            'text': 'Tom 在打篮球'
        }, {
            'text': 'a v s e c s f e f g a a a  '
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'that is a green tree'
        }]
        tgt_list = [{
            'text': 'Tom is playing piano.'
        }, {
            'text': 'Tom plays piano.'
        }, {
            'text': 'Tom played piano.'
        }, {
            'text': 'I play piano.'
        }, {
            'text': 'to play piano.'
        }]
        dataset = Dataset.from_list(ds_list)
        op = TextActionFilter(lang='en')
        self._run_text_action_filter(dataset, tgt_list, op, ['text'])

    def test_zh_text_case(self):

        ds_list = [{
            'text': '小明在 弹奏钢琴'
        }, {
            'text': 'Tom is playing 篮球'
        }, {
            'text': '上上下下左左右右'
        }, {
            'text': 'Tom在打篮球'
        }, {
            'text': '我有一只猫，它是一只猫'
        }]
        tgt_list = [{'text': '小明在 弹奏钢琴'}, {'text': 'Tom在打篮球'}]
        dataset = Dataset.from_list(ds_list)
        op = TextActionFilter(lang='zh')
        self._run_text_action_filter(dataset, tgt_list, op, ['text'])

    def test_image_text_case(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}小猫咪正在睡觉。{SpecialTokens.eoc}',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}小猫咪',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}背影{SpecialTokens.eoc}',
            'images': [self.img3_path]
        }, {
            'text': '雨中行走的女人背影',
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.image}小猫咪正在睡觉。{SpecialTokens.eoc}',
            'images': [self.cat_path]
        }, {
            'text': '雨中行走的女人背影',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = TextActionFilter(lang='zh')
        self._run_text_action_filter(dataset, tgt_list, op, ['text', 'images'])


if __name__ == '__main__':
    unittest.main()
