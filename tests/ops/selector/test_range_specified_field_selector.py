import unittest

from datasets import Dataset

from data_juicer.ops.selector.range_specified_field_selector import \
    RangeSpecifiedFieldSelector
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RangeSpecifiedFieldSelectorTest(DataJuicerTestCaseBase):

    def _run_range_selector(self, dataset: Dataset, target_list, op):
        dataset = op.process(dataset)
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['text'])
        target_list = sorted(target_list, key=lambda x: x['text'])
        self.assertEqual(res_list, target_list)

    def test_percentile_select(self):
        ds_list = [{
            'text': 'Today is Sun',
            'count': 101,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 34
                    },
                    'count': 5
                }
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'count': 16,
            'meta': {
                'suffix': '.docx',
                'key1': {
                    'key2': {
                        'count': 243
                    },
                    'count': 63
                }
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'count': 162,
            'meta': {
                'suffix': '.txt',
                'key1': {
                    'key2': {
                        'count': None
                    },
                    'count': 23
                }
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'count': None,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }, {
            'text': '他的英文名字叫Harry Potter',
            'count': 88,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 551
                    },
                    'count': 78
                }
            }
        }, {
            'text': '这是一个测试',
            'count': None,
            'meta': {
                'suffix': '.py',
                'key1': {
                    'key2': {
                        'count': 89
                    },
                    'count': 3
                }
            }
        }, {
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 67
                }
            }
        }, {
            'text': 'emoji表情测试下😊，😸31231\n',
            'count': 2,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 32
                }
            }
        }, {
            'text': 'a=1\nb\nc=1+2+3+5\nd=6',
            'count': 178,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 33
                    },
                    'count': 33
                }
            }
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言',
            'count': 666,
            'meta': {
                'suffix': '.xml',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }]
        tgt_list = [{
            'text': 'a v s e c s f e f g a a a  ',
            'count': 16,
            'meta': {
                'suffix': '.docx',
                'key1': {
                    'key2': {
                        'count': 243
                    },
                    'count': 63
                }
            }
        }, {
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 67
                }
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = RangeSpecifiedFieldSelector(field_key='meta.key1.count',
                                        lower_percentile=0.78,
                                        upper_percentile=0.9,
                                        lower_rank=5,
                                        upper_rank=10)
        self._run_range_selector(dataset, tgt_list, op)

    def test_rank_select(self):
        ds_list = [{
            'text': 'Today is Sun',
            'count': 101,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 34
                    },
                    'count': 5
                }
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'count': 16,
            'meta': {
                'suffix': '.docx',
                'key1': {
                    'key2': {
                        'count': 243
                    },
                    'count': 63
                }
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'count': 162,
            'meta': {
                'suffix': '.txt',
                'key1': {
                    'key2': {
                        'count': None
                    },
                    'count': 23
                }
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'count': None,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }, {
            'text': '他的英文名字叫Harry Potter',
            'count': 88,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 551
                    },
                    'count': 78
                }
            }
        }, {
            'text': '这是一个测试',
            'count': None,
            'meta': {
                'suffix': '.py',
                'key1': {
                    'key2': {
                        'count': 89
                    },
                    'count': 3
                }
            }
        }, {
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 67
                }
            }
        }, {
            'text': 'emoji表情测试下😊，😸31231\n',
            'count': 2,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 32
                }
            }
        }, {
            'text': 'a=1\nb\nc=1+2+3+5\nd=6',
            'count': 178,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 33
                    },
                    'count': 33
                }
            }
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言',
            'count': 666,
            'meta': {
                'suffix': '.xml',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }]
        tgt_list = [{
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 67
                }
            }
        }, {
            'text': 'emoji表情测试下😊，😸31231\n',
            'count': 2,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 32
                }
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = RangeSpecifiedFieldSelector(field_key='meta.key1.key2.count',
                                        lower_percentile=0.3,
                                        upper_percentile=1.0,
                                        lower_rank=7,
                                        upper_rank=9)
        self._run_range_selector(dataset, tgt_list, op)

    def test_percentile_rank_select(self):
        ds_list = [{
            'text': 'Today is Sun',
            'count': 101,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 34
                    },
                    'count': 5
                }
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'count': 16,
            'meta': {
                'suffix': '.docx',
                'key1': {
                    'key2': {
                        'count': 243
                    },
                    'count': 63
                }
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'count': 162,
            'meta': {
                'suffix': '.txt',
                'key1': {
                    'key2': {
                        'count': None
                    },
                    'count': 23
                }
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'count': None,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 18
                    },
                    'count': 48
                }
            }
        }, {
            'text': '他的英文名字叫Harry Potter',
            'count': 88,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 551
                    },
                    'count': 78
                }
            }
        }, {
            'text': '这是一个测试',
            'count': None,
            'meta': {
                'suffix': '.py',
                'key1': {
                    'key2': {
                        'count': 89
                    },
                    'count': 3
                }
            }
        }, {
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 67
                }
            }
        }, {
            'text': 'emoji表情测试下😊，😸31231\n',
            'count': 2,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 32
                }
            }
        }, {
            'text': 'a=1\nb\nc=1+2+3+5\nd=6',
            'count': 178,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': 33
                    },
                    'count': 33
                }
            }
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言',
            'count': 666,
            'meta': {
                'suffix': '.xml',
                'key1': {
                    'key2': {
                        'count': 2
                    },
                    'count': 48
                }
            }
        }]
        tgt_list = [{
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 67
                }
            }
        }, {
            'text': 'emoji表情测试下😊，😸31231\n',
            'count': 2,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': 354.32
                    },
                    'count': 32
                }
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = RangeSpecifiedFieldSelector(field_key='meta.key1.key2.count',
                                        lower_percentile=0.7,
                                        upper_percentile=1.0,
                                        lower_rank=3,
                                        upper_rank=9)
        self._run_range_selector(dataset, tgt_list, op)

    def test_list_select(self):
        ds_list = [{
            'text': 'Today is Sun',
            'count': 101,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': [34.0]
                    },
                    'count': 5
                }
            }
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            'count': 16,
            'meta': {
                'suffix': '.docx',
                'key1': {
                    'key2': {
                        'count': [243.0]
                    },
                    'count': 63
                }
            }
        }, {
            'text': '中文也是一个字算一个长度',
            'count': 162,
            'meta': {
                'suffix': '.txt',
                'key1': {
                    'key2': {
                        'count': []
                    },
                    'count': 23
                }
            }
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            'count': None,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': None
                    },
                    'count': 48
                }
            }
        }, {
            'text': '他的英文名字叫Harry Potter',
            'count': 88,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': [551.0]
                    },
                    'count': 78
                }
            }
        }, {
            'text': '这是一个测试',
            'count': None,
            'meta': {
                'suffix': '.py',
                'key1': {
                    'key2': {
                        'count': [89.0]
                    },
                    'count': 3
                }
            }
        }, {
            'text': '我出生于2023年12月15日',
            'count': None,
            'meta': {
                'suffix': '.java',
                'key1': {
                    'key2': {
                        'count': [354.32]
                    },
                    'count': 67
                }
            }
        }, {
            'text': 'emoji表情测试下😊，😸31231\n',
            'count': 2,
            'meta': {
                'suffix': '.html',
                'key1': {
                    'key2': {
                        'count': [354.32]
                    },
                    'count': 32
                }
            }
        }, {
            'text': 'a=1\nb\nc=1+2+3+5\nd=6',
            'count': 178,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': [33.0, 33.0]
                    },
                    'count': 33
                }
            }
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言',
            'count': 666,
            'meta': {
                'suffix': '.xml',
                'key1': {
                    'key2': {
                        'count': [2.0, 2.0]
                    },
                    'count': 48
                }
            }
        }]
        tgt_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6',
            'count': 178,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': [33.0, 33.0]
                    },
                    'count': 33
                }
            }
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言',
            'count': 666,
            'meta': {
                'suffix': '.xml',
                'key1': {
                    'key2': {
                        'count': [2.0, 2.0]
                    },
                    'count': 48
                }
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = RangeSpecifiedFieldSelector(field_key='meta.key1.key2.count',
                                        lower_percentile=0.0,
                                        upper_percentile=0.5,
                                        lower_rank=2,
                                        upper_rank=4)
        self._run_range_selector(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
