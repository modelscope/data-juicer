# maximum_line_length_filter

Filter to keep samples with a maximum line length within a specified range.

This operator filters out samples based on the length of their longest line. It retains
samples where the maximum line length is within the specified `min_len` and `max_len`
range. The maximum line length is computed by splitting the text into lines and
measuring the length of each line. If the context is provided, it uses precomputed lines
stored under the key 'lines' in the context. The maximum line length is cached in the
'max_line_length' field of the stats.

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