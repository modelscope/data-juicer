import unittest
import json

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.query_sentiment_detection_mapper import QuerySentimentDetectionMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.common_utils import nested_access

class TestQuerySentimentDetectionMapper(DataJuicerTestCaseBase):

    hf_model = '/mnt/workspace/shared/checkpoints/huggingface/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
    zh_to_en_hf_model = '/mnt/workspace/shared/checkpoints/huggingface/Helsinki-NLP/opus-mt-zh-en'

    def _run_op(self, op, samples, label_key, targets):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)

        for sample, target in zip(dataset, targets):
            label = nested_access(sample[Fields.meta], label_key)
            self.assertEqual(label, target)
        
    def test_default(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': '嗯嗯'
        },{
            'query': '没有希望。'
        },
        ]
        targets = ['positive', 'neutral', 'negative']

        op = QuerySentimentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
        )
        self._run_op(op, samples, MetaKeys.query_sentiment_label, targets)
    
    def test_no_zh_to_en(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': 'That is great!'
        }
        ]
        targets = ['neutral', 'positive']

        op = QuerySentimentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = None,
        )
        self._run_op(op, samples, MetaKeys.query_sentiment_label, targets)


if __name__ == '__main__':
    unittest.main()
