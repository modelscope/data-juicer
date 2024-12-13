import unittest
import json

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.query_sentiment_intensity_mapper import QuerySentimentLabelMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.common_utils import nested_access

class TestQuerySentimentLabelMapper(DataJuicerTestCaseBase):

    hf_model = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
    zh_to_en_hf_model = 'Helsinki-NLP/opus-mt-zh-en'

    def _run_op(self, op, samples, intensity_key, targets):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)

        for sample, target in zip(dataset, targets):
            intensity = nested_access(sample[Fields.meta], intensity_key)
            self.assertEqual(intensity, target)
        
    def test_default(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': '嗯嗯'
        },{
            'query': '没有希望。'
        },
        ]
        targets = [1, 0, -1]

        op = QuerySentimentLabelMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
        )
        self._run_op(op, samples, MetaKeys.query_sentiment_intensity, targets)
    
    def test_no_zh_to_en(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': 'That is great!'
        }
        ]
        targets = [0, 1]

        op = QuerySentimentLabelMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = None,
        )
        self._run_op(op, samples, MetaKeys.query_sentiment_intensity, targets)
    
    def test_reset_map1(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': '嗯嗯'
        },{
            'query': '没有希望。'
        },
        ]
        targets = [2, 0, -2]

        op = QuerySentimentLabelMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
            label_to_intensity = {
                'negative': -2,
                'neutral': 0,
                'positive': 2,
            }
        )
        self._run_op(op, samples, MetaKeys.query_sentiment_intensity, targets)

    def test_reset_map2(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': '嗯嗯'
        },{
            'query': '没有希望。'
        },
        ]
        targets = ['positive', 'neutral', 'negative']

        op = QuerySentimentLabelMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
            label_to_intensity = {}
        )
        self._run_op(op, samples, MetaKeys.query_sentiment_intensity, targets)

if __name__ == '__main__':
    unittest.main()
