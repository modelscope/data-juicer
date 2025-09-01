# maximum_line_length_filter

Filter to keep samples with a maximum line length within a specified range.

This operator filters out samples based on the length of their longest line. It retains samples where the maximum line length is within the specified `min_len` and `max_len` range. The maximum line length is computed by splitting the text into lines and measuring the length of each line. If the context is provided, it uses precomputed lines stored under the key 'lines' in the context. The maximum line length is cached in the 'max_line_length' field of the stats.

筛选器将最大行长度的样本保持在指定范围内。

此运算符根据样本最长行的长度过滤样本。它保留最大行长度在指定的 “min_len” 和 “max_len” 范围内的样本。通过将文本拆分为行并测量每行的长度来计算最大行长度。如果提供了上下文，则它使用存储在上下文中的键 “lines” 下的预先计算的行。最大行长度缓存在stats的 'max_line_length' 字段中。

Type 算子类型: **filter**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_len` | <class 'int'> | `10` | The min filter length in this op, samples will |
| `max_len` | <class 'int'> | `9223372036854775807` | The max filter length in this op, samples will |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/maximum_line_length_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_maximum_line_length_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)