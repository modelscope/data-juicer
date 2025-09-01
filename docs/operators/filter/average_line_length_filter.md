# average_line_length_filter

Filter to keep samples with average line length within a specific range.

This operator filters out samples based on their average line length. It keeps samples where the average line length is between the specified minimum and maximum values. The average line length is calculated as the total text length divided by the number of lines. If the context is provided, it uses precomputed lines from the context. The computed average line length is stored in the 'avg_line_length' key in the stats field.

过滤器，以保持平均线长度在特定范围内的样本。

此运算符根据平均线长度过滤出样本。它会保留平均线长度介于指定的最小值和最大值之间的样本。平均线长度计算为总文本长度除以行数。如果提供了上下文，则它使用来自上下文的预先计算的行。计算的平均线路长度存储在stats字段中的 “avg_line_length” 关键字中。

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
- [source code 源代码](../../../data_juicer/ops/filter/average_line_length_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_average_line_length_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)