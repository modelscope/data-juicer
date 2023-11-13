import os
import unittest

from datasets import Dataset

from data_juicer.ops.filter.clip_similarity_filter import (
    ClipSimilarityFilter, SpecialTokens)
from data_juicer.utils.constant import Fields


class ClipSimilarityFilterTest(unittest.TestCase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    cat_path = os.path.join(data_path, 'cat.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
    hf_clip = 'openai/clip-vit-base-patch32'

    def _run_filter(self, dataset: Dataset, target_list, op):
         
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
    
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['text', 'images'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_no_eoc_special_token(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo of a dog',
            'images': [self.cat_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = ClipSimilarityFilter(hf_clip=self.hf_clip,
                                  reduce_mode='avg',
                                  any_or_all='any',
                                  min_ratio=0.2,
                                  max_ratio=0.9)
        self._run_filter(dataset, tgt_list, op)

    def test_eoc_special_token(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a photo of a cat{SpecialTokens.eoc}',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo of a dog',
            'images': [self.cat_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a photo of a cat{SpecialTokens.eoc}',
            'images': [self.cat_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = ClipSimilarityFilter(hf_clip=self.hf_clip,
                                  reduce_mode='avg',
                                  any_or_all='any',
                                  min_ratio=0.2,
                                  max_ratio=0.9)
        self._run_filter(dataset, tgt_list, op)

    def test_keep_any(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a photo of a cat {SpecialTokens.eoc} '
            f'{SpecialTokens.image}a photo of a dog {SpecialTokens.eoc}',
            'images': [self.cat_path, self.cat_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a photo of a cat {SpecialTokens.eoc} '
            f'{SpecialTokens.image}a photo of a dog {SpecialTokens.eoc}',
            'images': [self.cat_path, self.cat_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ClipSimilarityFilter(hf_clip=self.hf_clip,
                                  reduce_mode='avg',
                                  any_or_all='any',
                                  min_ratio=0.2,
                                  max_ratio=0.9)
        self._run_filter(dataset, tgt_list, op)

    def test_keep_all(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a photo of a cat {SpecialTokens.eoc} '
            f'{SpecialTokens.image}a photo of a dog {SpecialTokens.eoc}',
            'images': [self.cat_path, self.cat_path]
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = ClipSimilarityFilter(hf_clip=self.hf_clip,
                                  reduce_mode='avg',
                                  any_or_all='all',
                                  min_ratio=0.2,
                                  max_ratio=0.9)
        self._run_filter(dataset, tgt_list, op)

    def test_reduce_avg(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat '
            f'{SpecialTokens.image} {SpecialTokens.eoc}',
            'images': [self.cat_path, self.img3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat '
            f'{SpecialTokens.image} {SpecialTokens.eoc}',
            'images': [self.cat_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ClipSimilarityFilter(hf_clip=self.hf_clip,
                                  reduce_mode='avg',
                                  any_or_all='any',
                                  min_ratio=0.2,
                                  max_ratio=0.9)
        self._run_filter(dataset, tgt_list, op)

    def xxtest_reduce_max(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat '
            f'{SpecialTokens.image} {SpecialTokens.eoc}',
            'images': [self.cat_path, self.img3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat '
            f'{SpecialTokens.image} {SpecialTokens.eoc}',
            'images': [self.cat_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ClipSimilarityFilter(hf_clip=self.hf_clip,
                                  reduce_mode='max',
                                  any_or_all='any',
                                  min_ratio=0.2,
                                  max_ratio=0.9)
        self._run_filter(dataset, tgt_list, op)

    def test_reduce_min(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat '
            f'{SpecialTokens.image} {SpecialTokens.eoc}',
            'images': [self.cat_path, self.img3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat '
            f'{SpecialTokens.image} {SpecialTokens.eoc}',
            'images': [self.cat_path, self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = ClipSimilarityFilter(hf_clip=self.hf_clip,
                                  reduce_mode='min',
                                  any_or_all='any',
                                  min_ratio=0.1,
                                  max_ratio=0.9)

        self._run_filter(dataset, tgt_list, op)

        op.min_ratio = 0.2
        self._run_filter(dataset, [], op)


if __name__ == '__main__':
    unittest.main()
