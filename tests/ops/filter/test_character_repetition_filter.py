import unittest

from datasets import Dataset

from data_juicer.ops.filter.character_repetition_filter import \
    CharacterRepetitionFilter
from data_juicer.utils.constant import Fields


class CharacterRepetitionFilterTest(unittest.TestCase):

    def _run_character_repetition_filter(self, dataset: Dataset, target_list,
                                         op):
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
            'text':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text': 'a v s e c s f e f g a a a a a a a a a a'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': '中文也是一个字算一个长度'
        }]
        tgt_list = [{
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': '中文也是一个字算一个长度'
        }]
        dataset = Dataset.from_list(ds_list)
        op = CharacterRepetitionFilter(rep_len=5,
                                       min_ratio=0.0,
                                       max_ratio=0.4)
        self._run_character_repetition_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
