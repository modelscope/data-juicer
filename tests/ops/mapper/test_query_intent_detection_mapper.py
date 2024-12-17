import unittest
import json

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.query_intent_detection_mapper import QueryIntentDetectionMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.common_utils import nested_access

class TestQueryIntentDetectionMapper(DataJuicerTestCaseBase):

    hf_model = '/mnt/workspace/shared/checkpoints/huggingface/bespin-global/klue-roberta-small-3i4k-intent-classification'
    zh_to_en_hf_model = '/mnt/workspace/shared/checkpoints/huggingface/Helsinki-NLP/opus-mt-zh-en'

    def _run_op(self, op, samples, label_key, targets):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)

        for sample, target in zip(dataset, targets):
            label = nested_access(sample[Fields.meta], label_key)
            print(label)
            # self.assertEqual(label, target)
        
    def test_default(self):
        
        samples = [{
            'query': '这样好吗？'
        },{
            'query': '站住！'
        },{
            'query': '今天阳光灿烂。'
        }
        ]
        targets = [1, 0, -1]

        op = QueryIntentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
        )
        self._run_op(op, samples, MetaKeys.query_intent_label, targets)
    
    def test_no_zh_to_en(self):
        
        samples = [{
            'query': '这样好吗？'
        },{
            'query': 'Is this okay?'
        }
        ]
        targets = [0, 1]

        op = QueryIntentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = None,
        )
        self._run_op(op, samples, MetaKeys.query_intent_label, targets)

if __name__ == '__main__':
    unittest.main()
