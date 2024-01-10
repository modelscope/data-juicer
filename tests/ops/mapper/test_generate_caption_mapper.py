import os
import unittest

from data_juicer.core.data import NestedDataset
from data_juicer.ops.mapper.generate_caption_mapper import \
    GenerateCaptionMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, SKIPPED_TESTS

# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class GenerateCaptionMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    cat_path = os.path.join(data_path, 'cat.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    hf_blip2 = 'Salesforce/blip2-opt-2.7b'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_blip2)

    def _run_mapper(self, dataset: NestedDataset, op, num_proc=1, caption_num=0):

        dataset = dataset.map(op.process, num_proc=num_proc)
        dataset_list = dataset.select_columns(column_names=['text']).to_list()
        # assert the caption is generated successfully in terms of not_none
        # as the generated content is not deterministic
        self.assertEqual(len(dataset_list), caption_num)

    def test_no_eoc_special_token(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 1
        dataset = NestedDataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2=self.hf_blip2,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, caption_num=len(dataset) * 2)

    def test_eoc_special_token(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat{SpecialTokens.eoc}',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella{SpecialTokens.eoc}',
            'images': [self.img3_path]
        }]
        caption_num = 1
        dataset = NestedDataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2=self.hf_blip2,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, caption_num=len(dataset) * 2)

    def test_multi_candidate_keep_random_any(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = NestedDataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2=self.hf_blip2,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, caption_num=len(dataset) * 2)

    def test_multi_candidate_keep_all(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = NestedDataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2=self.hf_blip2,
                                   caption_num=caption_num,
                                   keep_candidate_mode='all')
        self._run_mapper(dataset, op, caption_num=(1 + caption_num) * len(dataset))

    def test_multi_candidate_keep_similar_one(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = NestedDataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2=self.hf_blip2,
                                   caption_num=caption_num,
                                   keep_candidate_mode='similar_one_simhash')
        self._run_mapper(dataset, op, caption_num=len(dataset) * 2)

    def test_multi_process(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }] * 10
        caption_num = 1
        dataset = NestedDataset.from_list(ds_list)
        op = GenerateCaptionMapper(hf_blip2=self.hf_blip2,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, num_proc=4, caption_num=len(dataset) * 2)


if __name__ == '__main__':
    unittest.main()
