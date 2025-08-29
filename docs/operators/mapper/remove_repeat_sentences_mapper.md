# remove_repeat_sentences_mapper

Mapper to remove repeat sentences in text samples.

This operator processes text samples to remove duplicate sentences. It splits the text
into lines and then further splits each line into sentences. Sentences are considered
duplicates if they are identical after optional case normalization and special character
removal. The operator uses a hash set to track unique sentences. Sentences shorter than
`min_repeat_sentence_length` are not deduplicated. If `ignore_special_character` is
enabled, special characters (all except Chinese, letters, and numbers) are ignored when
checking for duplicates. The resulting text is reassembled with unique sentences.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `lowercase` | <class 'bool'> | `False` | Whether to convert sample text to lower case |
| `ignore_special_character` | <class 'bool'> | `True` | Whether to ignore special |
| `min_repeat_sentence_length` | <class 'int'> | `2` | Sentences shorter than this |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_repeat_sentences_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_repeat_sentences_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)