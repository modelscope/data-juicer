import unittest

from datasets import Dataset

from data_juicer.ops.selector.topk_specified_field_selector import \
    TopkSpecifiedFieldSelector


class TopkSpecifiedFieldSelectorTest(unittest.TestCase):

    def _run_topk_selector(self, dataset: Dataset, target_list, op):
        dataset = op.process(dataset)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_topratio_select(self):
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
        op = TopkSpecifiedFieldSelector(field_key='meta.key1.key2.count',
                                        top_ratio=0.2,
                                        topk=5,
                                        reverse=True)
        self._run_topk_selector(dataset, tgt_list, op)

    def test_topk_select(self):
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
        }]
        dataset = Dataset.from_list(ds_list)
        op = TopkSpecifiedFieldSelector(field_key='meta.key1.key2.count',
                                        top_ratio=0.5,
                                        topk=4,
                                        reverse=True)
        self._run_topk_selector(dataset, tgt_list, op)

    def test_reverse_select(self):
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
        }]
        dataset = Dataset.from_list(ds_list)
        op = TopkSpecifiedFieldSelector(field_key='meta.key1.key2.count',
                                        top_ratio=0.5,
                                        topk=3,
                                        reverse=False)
        self._run_topk_selector(dataset, tgt_list, op)

    def test_str_select(self):
        ds_list = [{
            'text': 'Today is Sun',
            'count': 101,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': '34'
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
                        'count': '243'
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
                        'count': ''
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
                        'count': '551'
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
                        'count': '89'
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
                        'count': '354.32'
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
                        'count': '354.32'
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
                        'count': '33'
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
                        'count': '2'
                    },
                    'count': 48
                }
            }
        }]
        tgt_list = [{
            'text': '使用片段分词器对每个页面进行分词，使用语言',
            'count': 666,
            'meta': {
                'suffix': '.xml',
                'key1': {
                    'key2': {
                        'count': '2'
                    },
                    'count': 48
                }
            }
        }, {
            'text': 'a=1\nb\nc=1+2+3+5\nd=6',
            'count': 178,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': '33'
                    },
                    'count': 33
                }
            }
        }, {
            'text': 'Today is Sun',
            'count': 101,
            'meta': {
                'suffix': '.pdf',
                'key1': {
                    'key2': {
                        'count': '34'
                    },
                    'count': 5
                }
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = TopkSpecifiedFieldSelector(field_key='meta.key1.key2.count',
                                        top_ratio=0.5,
                                        topk=3,
                                        reverse=False)
        self._run_topk_selector(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
