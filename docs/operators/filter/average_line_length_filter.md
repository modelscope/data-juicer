# average_line_length_filter

Filter to keep samples with average line length within a specific range.

This operator filters out samples based on their average line length. It keeps samples
where the average line length is between the specified minimum and maximum values. The
average line length is calculated as the total text length divided by the number of
lines. If the context is provided, it uses precomputed lines from the context. The
computed average line length is stored in the 'avg_line_length' key in the stats field.

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