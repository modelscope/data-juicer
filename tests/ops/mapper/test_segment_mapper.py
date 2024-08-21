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

        ds_list = [{
            'images': [img1_path]
        }, {
            'images': [img2_path]
        }]

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        print(dataset["bboxes"])

    def test_segment_mapper(self):
        self._run_segment_mapper()



if __name__ == '__main__':
    unittest.main()