# remove_table_text_mapper

Mapper to remove table texts from text samples.

This operator uses regular expressions to identify and remove tables from the text. It
targets tables with a specified range of columns, defined by the minimum and maximum
number of columns. The operator iterates over each sample, applying the regex pattern to
remove tables that match the column criteria. The processed text, with tables removed,
is then stored back in the sample. This operation is batched for efficiency.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_col` | typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=2), Le(le=20)])] | `2` | The min number of columns of table to remove. |
| `max_col` | typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=2), Le(le=20)])] | `20` | The max number of columns of table to remove. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_table_text_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_table_text_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)