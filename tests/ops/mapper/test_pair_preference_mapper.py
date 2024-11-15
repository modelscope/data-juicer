import unittest

from loguru import logger

from data_juicer.ops.mapper.pair_preference_mapper import PairPreferenceMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)


# Skip tests for this OP because the API call is not configured yet.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class PairPreferenceMapperTest(DataJuicerTestCaseBase):

    def _run_op(self, api_model, response_path=None):

        op = PairPreferenceMapper(api_model=api_model,
                                  response_path=response_path)

        reference = '王八十娘：小远城王八十的娘亲，李莲花刚到小远城时被方多病偷掉钱袋找小乞丐问路时，刚好发现王八十娘被另一个小乞丐撞到便将她扶起，结识了王八十。\n朴二黄：灵山派管家，方多病小厮旺福的父亲。真实身份是金鸳盟的奔雷手辛雷，离开金鸳盟后，用假名朴二黄在灵山派当管家。因害怕王青山看穿他的身份，设计杀死了灵山派的王青山。被捕后识破了李莲花的真实身份，最后在攻击李莲花的时候被方多病情急之下杀死。'

        samples = [{
            'text': reference,
            'query': '李莲花，你认识方多病吗?',
            'response': '方多病啊，那可是我的好友。'
        }]

        for sample in samples:
            result = op.process(sample)
            logger.info(f'Output results: {result}')
            self.assertNotEqual(result['rejected_response'], '')
            self.assertNotEqual(result['reason'], '')

    def test(self):
        # before runing this test, set below environment variables:
        # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
        # export OPENAI_API_KEY=your_key
        self._run_op('qwen2.5-72b-instruct')


if __name__ == '__main__':
    unittest.main()
