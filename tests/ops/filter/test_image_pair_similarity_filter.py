import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.image_pair_similarity_filter import ImagePairSimilarityFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImagePairSimilarityFilterTest(DataJuicerTestCaseBase):

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
        dataset = dataset.select_columns(column_names=['text', 'images'])
        res_list = dataset.to_list()
        print(res_list)

    def test_no_eoc_special_token(self):

        ds_list = [{
            'text': 'image pair 1',
            'images': ["./0.jpg", "./1.jpg"]
        }, {
            'text': 'image pair 2',
            'images': ["./crayon.jpg", "./ipod.jpg"]
        }]


        dataset = Dataset.from_list(ds_list)
        op = ImagePairSimilarityFilter(hf_clip="clip-vit-base-patch32",
                                       any_or_all='any',
                                       min_score=0.85,
                                       max_score=0.98)
        self._run_filter(dataset, op)


if __name__ == '__main__':
    unittest.main()
