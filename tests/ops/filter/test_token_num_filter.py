import unittest

from datasets import Dataset

from data_juicer.ops.filter.token_num_filter import TokenNumFilter
from data_juicer.utils.constant import Fields, StatsKeys


class WordNumFilterTest(unittest.TestCase):

    def test_token_num(self):
        src = [
            {"text": "Today is Sunday and it's a happy day!"},
            {"text": "Do you need a cup of coffee?"},
            {"text": "你好，请问你是谁"},
            {"text": "Sur la plateforme MT4, plusieurs manières d'accéder à "
                     "ces fonctionnalités sont conçues simultanément."},
            {"text": "欢迎来到阿里巴巴！"},
            {"text": "This paper proposed a novel method on LLM pretraining."},
        ]
        tgt = [
            10, 8, 9, 31, 14, 12,
        ]
        ds = Dataset.from_list(src)
        op = TokenNumFilter()
        ds = ds.add_column(Fields.stats, [{}] * len(ds))
        ds = ds.map(op.compute_stats)
        stats = ds[Fields.stats]
        self.assertEqual([item[StatsKeys.num_token] for item in stats], tgt)

    def test_token_num_filter(self):
        src = [
            {"text": "Today is Sunday and it's a happy day!"},
            {"text": "Do you need a cup of coffee?"},
            {"text": "你好，请问你是谁"},
            {"text": "Sur la plateforme MT4, plusieurs manières d'accéder à "
                     "ces fonctionnalités sont conçues simultanément."},
            {"text": "欢迎来到阿里巴巴！"},
            {"text": "This paper proposed a novel method on LLM pretraining."},
        ]
        tgt = [
            {"text": "Today is Sunday and it's a happy day!"},
            {"text": "欢迎来到阿里巴巴！"},
            {"text": "This paper proposed a novel method on LLM pretraining."},
        ]
        ds = Dataset.from_list(src)
        tgt = Dataset.from_list(tgt)
        op = TokenNumFilter(min_num=10, max_num=20)
        ds = ds.add_column(Fields.stats, [{}] * len(ds))
        ds = ds.map(op.compute_stats)
        ds = ds.filter(op.process)
        self.assertEqual(ds['text'], tgt['text'])


if __name__ == '__main__':
    unittest.main()
