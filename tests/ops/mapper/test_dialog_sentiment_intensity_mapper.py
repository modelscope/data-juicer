import unittest
import json

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.dialog_sentiment_intensity_mapper import DialogSentimentIntensityMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)
from data_juicer.utils.constant import MetaKeys
from data_juicer.utils.common_utils import nested_access

# Skip tests for this OP.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class TestDialogSentimentIntensityMapper(DataJuicerTestCaseBase):


    def _run_op(self, op):

        samples = [{
            'history': [
                (
                    '李莲花有口皆碑',
                    '「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。'
                ),
                (
                    '是的，你确实是一个普通大夫，没什么值得夸耀的。',
                    '「委屈」你这话说的，我也是尽心尽力治病救人了。'
                ),
                (
                    '你自己说的呀，我现在说了，你又不高兴了。',
                    'or of of of of or or and or of of of of of of of,,, '
                ),
                (
                    '你在说什么我听不懂。',
                    '「委屈」我也没说什么呀，就是觉得你有点冤枉我了'
                )
            ]
        }]

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
        analysis_list = nested_access(dataset, MetaKeys.sentiment_analysis)
        intensity_list = nested_access(dataset, MetaKeys.sentiment_intensity)

        for analysis, intensity in zip(analysis_list, intensity_list):
            logger.info(f'分析：{analysis}')
            logger.info(f'情绪：{intensity}')
        
    def default_test(self):
        # before runing this test, set below environment variables:
        # export OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
        # export OPENAI_API_KEY=your_key
        op = DialogSentimentIntensityMapper(api_model='qwen2.5-72b-instruct')
        self._run_op(op)
    
    def max_round_test(self):
        op = DialogSentimentIntensityMapper(api_model='qwen2.5-72b-instruct')
        self._run_op(op)


if __name__ == '__main__':
    unittest.main()
