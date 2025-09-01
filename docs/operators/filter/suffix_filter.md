# suffix_filter

Filter to keep samples with specified suffix.

This operator retains samples that have a suffix matching any of the provided suffixes. If no suffixes are specified, all samples are kept. The key metric 'keep' is computed based on whether the sample's suffix matches the specified list. The 'suffix' field of each sample is checked against the list of allowed suffixes. If the suffix matches, the sample is kept; otherwise, it is filtered out.

过滤器以保留具有指定后缀的样本。

此运算符保留具有与所提供的任何后缀匹配的后缀的样本。如果未指定后缀，则保留所有样本。根据样本的后缀是否与指定列表匹配来计算关键指标 “keep”。对照允许的后缀列表检查每个样本的 “后缀” 字段。如果后缀匹配，则保留样本; 否则，将其过滤掉。

Type 算子类型: **filter**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `suffixes` | typing.Union[str, typing.List[str]] | `[]` | the suffix of text that will be keep. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/suffix_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_suffix_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)