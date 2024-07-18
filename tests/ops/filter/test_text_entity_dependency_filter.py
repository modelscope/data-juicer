import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.text_entity_dependency_filter import \
    TextEntityDependencyFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TextEntityDependencyFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    cat_path = os.path.join(data_path, 'cat.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    def _run_text_entity_denpendency_filter(self, dataset: Dataset,
                                            target_list, op, column_names):
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
            'text': 'She is smiling.'
        }, {
            'text': 'Tom is playing piano.'
        }, {
            'text': 'piano.'
        }, {
            'text': 'a green tree',
        }, {
            'text': 'tree',
        }, {
            'text': 'book. mountain. star. potato',
        }]
        tgt_list = [{
            'text': 'She is smiling.'
        }, {
            'text': 'Tom is playing piano.'
        }, {
            'text': 'a green tree',
        }]
        dataset = Dataset.from_list(ds_list)
        op = TextEntityDependencyFilter(lang='en', any_or_all='any')
        self._run_text_entity_denpendency_filter(dataset, tgt_list, op,
                                                 ['text'])

    def test_zh_text_case(self):

        ds_list = [{
            'text': '她在笑'
        }, {
            'text': '枯藤老树昏鸦'
        }, {
            'text': '上上下下左左右右'
        }, {
            'text': '一只会上树的猫'
        }, {
            'text': '猫'
        }, {
            'text': '书。山。星星。土豆。'
        }]
        tgt_list = [{'text': '她在笑'}, {'text': '枯藤老树昏鸦'}, {'text': '一只会上树的猫'}]
        dataset = Dataset.from_list(ds_list)
        op = TextEntityDependencyFilter(lang='zh', any_or_all='all')
        self._run_text_entity_denpendency_filter(dataset, tgt_list, op,
                                                 ['text'])

    def test_image_text_case(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}三只缩成一团的小猫咪。{SpecialTokens.eoc}',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}猫咪',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}背影{SpecialTokens.eoc}',
            'images': [self.img3_path]
        }, {
            'text': '撑着伞的女人背影',
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.image}三只缩成一团的小猫咪。{SpecialTokens.eoc}',
            'images': [self.cat_path]
        }, {
            'text': '撑着伞的女人背影',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = TextEntityDependencyFilter(lang='zh', any_or_all='any')
        self._run_text_entity_denpendency_filter(dataset, tgt_list, op,
                                                 ['text', 'images'])


if __name__ == '__main__':
    unittest.main()
