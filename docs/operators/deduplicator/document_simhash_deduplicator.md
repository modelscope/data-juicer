# document_simhash_deduplicator

Deduplicates samples at the document level using SimHash.

This operator computes SimHash values for each sample and removes duplicates based on a
specified Hamming distance threshold. It supports different tokenization methods:
'space', 'punctuation', and 'character'. The SimHash is computed over shingles of a
given window size, and the deduplication process clusters similar documents and retains
only one from each cluster. The default mode converts text to lowercase and can ignore
specific patterns. The key metric, Hamming distance, is used to determine similarity
between SimHash values. Important notes:
- The `ignore_pattern` parameter can be used to exclude certain substrings during
SimHash computation.
- For punctuation-based tokenization, the `ignore_pattern` should not include
punctuations to avoid conflicts.
- The `hamming_distance` must be less than the number of blocks (`num_blocks`).
- Only the first sample in each cluster is retained by default.

Type 算子类型: **deduplicator**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tokenization` | <class 'str'> | `'space'` |  |
| `window_size` | typing.Annotated[int, Gt(gt=0)] | `6` | window size of shingling |
| `lowercase` | <class 'bool'> | `True` | whether to convert text to lower case first |
| `ignore_pattern` | typing.Optional[str] | `None` | whether to ignore sub-strings with |
| `num_blocks` | typing.Annotated[int, Gt(gt=0)] | `6` | number of blocks in simhash computing |
| `hamming_distance` | typing.Annotated[int, Gt(gt=0)] | `4` | the max hamming distance threshold in |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/deduplicator/document_simhash_deduplicator.py)
- [unit test 单元测试](../../../tests/ops/deduplicator/test_document_simhash_deduplicator.py)
- [Return operator list 返回算子列表](../../Operators.md)