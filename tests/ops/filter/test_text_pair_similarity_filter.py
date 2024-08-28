import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.text_pair_similarity_filter import TextPairSimilarityFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)


class TextPairSimilarityFilterTest(DataJuicerTestCaseBase):

    hf_clip = 'openai/clip-vit-base-patch32'

    
    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_clip)

    def _run_filter(self, dataset: Dataset, op, num_proc=1):

        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)

        dataset = dataset.map(op.compute_stats,
                              num_proc=num_proc,
                              with_rank=True)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.select_columns(column_names=['text', 'target_text'])
        res_list = dataset.to_list()
        print(res_list)

    def test_no_eoc_special_token(self):

        ds_list = [{
            'target_text': 'a lovely cat',
            'text': 'a lovely cat',
        }, {
            'target_text': 'a lovely cat',
            'text': 'a cute cat',
        }, {
            'target_text': 'a lovely cat',
            'text': 'a black dog',
        }]


        dataset = Dataset.from_list(ds_list)
        op = TextPairSimilarityFilter(hf_clip=self.hf_clip,
                                       any_or_all='any',
                                       min_score=0.1,
                                       max_score=0.85)
        self._run_filter(dataset, op)


if __name__ == '__main__':
    unittest.main()
