# token_num_filter

Filter to keep samples with a total token number within a specified range.

This operator uses a Hugging Face tokenizer to count the number of tokens in each sample. It keeps samples where the token count is between the minimum and maximum thresholds. The token count is stored in the 'num_token' field of the sample's stats. If the token count is not already computed, it will be calculated using the specified tokenizer.

筛选器将总令牌数的样本保留在指定范围内。

此运算符使用拥抱面标记器来计算每个样本中的标记数。它保留令牌计数在最小和最大阈值之间的样本。令牌计数存储在样本统计信息的 “num_token” 字段中。如果尚未计算令牌计数，则将使用指定的令牌发生器计算令牌计数。

Type 算子类型: **filter**

Tags 标签: cpu, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_tokenizer` | <class 'str'> | `'EleutherAI/pythia-6.9b-deduped'` | the tokenizer name of Hugging Face tokenizers. |
| `min_num` | <class 'int'> | `10` | The min filter token number in this op, samples |
| `max_num` | <class 'int'> | `9223372036854775807` | The max filter token number in this op, samples |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/token_num_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_token_num_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)