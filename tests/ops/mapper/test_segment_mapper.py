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
        
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
        img2_path = os.path.join(data_path, 'img2.jpg')
        img3_path = os.path.join(data_path, 'img3.jpg')
        img5_path = os.path.join(data_path, 'img5.jpg')

        ds_list = [{
            'images': [img2_path, img3_path]
        }, {
            'images': [img5_path]
        }]


        for sample in ds_list:
            result = op.process(sample)
            print(f'Output results: {result}')


    def test_segment_mapper(self):
        self._run_segment_mapper()



if __name__ == '__main__':
    unittest.main()