# text_chunk_mapper

Split input text into chunks based on specified criteria.

- Splits the input text into multiple chunks using a specified maximum length and a split pattern.
- If `max_len` is provided, the text is split into chunks with a maximum length of `max_len`.
- If `split_pattern` is provided, the text is split at occurrences of the pattern. If the length exceeds `max_len`, it will force a cut.
- The `overlap_len` parameter specifies the overlap length between consecutive chunks if the split does not occur at the pattern.
- Uses a Hugging Face tokenizer to calculate the text length in tokens if a tokenizer name is provided; otherwise, it uses the string length.
- Caches the following stats: 'chunk_count' (number of chunks generated for each sample).
- Raises a `ValueError` if both `max_len` and `split_pattern` are `None` or if `overlap_len` is greater than or equal to `max_len`.

根据指定的条件将输入文本拆分为块。

- 使用指定的最大长度和拆分模式将输入文本拆分为多个块。
- 如果提供了 'max_len'，则将文本拆分为最大长度为 'max_len' 的块。
- 如果提供了 “split_pattern”，则在出现该模式时拆分文本。如果长度超过 'max_len'，它将强制切割。
- 'overlap_len' 参数指定在模式处未发生分割的情况下连续块之间的重叠长度。
- 如果提供了标记器名称，则使用拥抱面标记器来计算标记中的文本长度; 否则，它使用字符串长度。
- 缓存以下统计信息: 'chunk_count' (为每个样本生成的块的数量)。
- 如果 “max_len” 和 “split_pattern” 均为 “none” 或 “overlap_len” 大于或等于 “max_len”，则引发 “valueerror”。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `max_len` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | Split text into multi texts with this max len if not |
| `split_pattern` | typing.Optional[str] | `'\n\n'` | Make sure split in this pattern if it is not None |
| `overlap_len` | typing.Annotated[int, Ge(ge=0)] | `0` | Overlap length of the split texts if not split in |
| `tokenizer` | typing.Optional[str] | `None` | The tokenizer name of Hugging Face tokenizers. |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/text_chunk_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_text_chunk_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)