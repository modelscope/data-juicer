# random_selector

Randomly selects a subset of samples from the dataset.

This operator randomly selects a subset of samples based on either a specified ratio or
a fixed number. If both `select_ratio` and `select_num` are provided, the one that
results in fewer samples is used. The selection is skipped if the dataset has only one
or no samples. The `random_sample` function is used to perform the actual sampling.

- `select_ratio`: The ratio of samples to select (0 to 1).
- `select_num`: The exact number of samples to select.
- If neither `select_ratio` nor `select_num` is set, the dataset remains unchanged.

Type 算子类型: **selector**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `select_ratio` | typing.Optional[typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=1)])]] | `None` | The ratio to select. When both |
| `select_num` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The number of samples to select. When both |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/selector/random_selector.py)
- [unit test 单元测试](../../../tests/ops/selector/test_random_selector.py)
- [Return operator list 返回算子列表](../../Operators.md)