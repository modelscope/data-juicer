import unittest
import json

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.query_topic_detection_mapper import QueryTopicDetectionMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.common_utils import nested_access

class TestQueryTopicDetectionMapper(DataJuicerTestCaseBase):

    hf_model = '/mnt/workspace/shared/checkpoints/huggingface/dstefa/roberta-base_topic_classification_nyt_news'
    zh_to_en_hf_model = '/mnt/workspace/shared/checkpoints/huggingface/Helsinki-NLP/opus-mt-zh-en'

    def _run_op(self, op, samples, label_key, targets):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)

        for sample, target in zip(dataset, targets):
            label = nested_access(sample[Fields.meta], label_key)
            self.assertEqual(label, target)
        
    def test_default(self):
        
        samples = [{
            'query': '今天火箭和快船的比赛谁赢了。'
        },{
            'query': '你最近身体怎么样。'
        }
        ]
        targets = ['Sports', 'Arts, Culture, and Entertainment', 'Health and Wellness']

        op = QueryTopicDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
        )
        self._run_op(op, samples, MetaKeys.query_topic_label, targets)
    
    def test_no_zh_to_en(self):
        
        samples = [{
            'query': '这样好吗？'
        },{
            'query': 'Is this okay?'
        }
        ]
        targets = ['Lifestyle and Fashion', 'Health and Wellness']

        op = QueryTopicDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = None,
        )
        self._run_op(op, samples, MetaKeys.query_topic_label, targets)

if __name__ == '__main__':
    unittest.main()
