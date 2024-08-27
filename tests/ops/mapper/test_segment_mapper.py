import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

from data_juicer.ops.mapper.segment_mapper import SegmentMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)



class SDXLPrompt2PromptMapperTest(DataJuicerTestCaseBase):

    text_key = 'text'

    def _run_segment_mapper(self, enable_vllm=False):
        op = SegmentMapper(
            fastsam_path='FastSAM-x.pt',
        )

        img1_path = './crayon.jpg'
        img2_path = './ipod.jpg'
        img3_path = './0_19_0_0.jpg'

        ds_list = [{
            'images': [img1_path, img3_path]
        }, {
            'images': [img2_path]
        }]


        for sample in ds_list:
            result = op.process(sample)
            print(f'Output results: {result}')


    def test_segment_mapper(self):
        self._run_segment_mapper()



if __name__ == '__main__':
    unittest.main()