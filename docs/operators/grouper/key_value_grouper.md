# key_value_grouper

Groups samples into batches based on values in specified keys.

This operator groups samples by the values of the given keys, which can be nested. If no
keys are provided, it defaults to using the text key. It uses a naive grouping strategy
to batch samples with identical key values. The resulting dataset is a list of batched
samples, where each batch contains samples that share the same key values. This is
useful for organizing data by specific attributes or features.

Type 算子类型: **grouper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `group_by_keys` | typing.Optional[typing.List[str]] | `None` | group samples according values in the keys. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/grouper/key_value_grouper.py)
- [unit test 单元测试](../../../tests/ops/grouper/test_key_value_grouper.py)
- [Return operator list 返回算子列表](../../Operators.md)