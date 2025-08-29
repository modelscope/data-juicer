# in_context_influence_filter

Filter to keep texts based on their in-context influence on a validation set.

This operator calculates the in-context influence of each sample by comparing
perplexities with and without the sample as context. The influence score is computed as
the ratio of these perplexities. If `valid_as_demo` is True, the score is L(A|Q) /
L(A|task_desc, Q_v, A_v, Q). Otherwise, it is L(A_v|Q) / L(A_v|task_desc, Q, A, Q_v).
The operator retains samples whose in-context influence score is within a specified
range. The in-context influence score is stored in the 'in_context_influence' field of
the sample's stats. The validation set must be prepared using the
`prepare_valid_feature` method if not provided during initialization.

Type 算子类型: **filter**

Tags 标签: cpu, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `valid_dataset` | typing.Optional[typing.List[typing.Dict]] | `None` | The dataset to use for validation. |
| `task_desc` | <class 'str'> | `None` | The description of the validation task. |
| `valid_as_demo` | <class 'bool'> | `False` | If true, score =  L(A|Q) / L(A|task_desc, Q_v, A_v, Q); |
| `n_shot` | typing.Optional[int] | `None` | The number of shots in validation. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/in_context_influence_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_in_context_influence_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)