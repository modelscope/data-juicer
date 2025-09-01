# general_field_filter

Filter to keep samples based on a general field filter condition.

The filter condition is a string that can include logical operators (and/or) and chain comparisons. For example: "10 < num <= 30 and text != 'nothing here' and __dj__meta__.a == 3". The condition is evaluated for each sample, and only samples that meet the condition are kept. The result of the filter condition is stored in the sample's stats under the key 'general_field_filter_condition'. If the filter condition is empty or already computed, the sample is not re-evaluated.

根据常规字段筛选条件保留样本。

过滤条件是可以包括逻辑运算符 (and/or) 和链比较的字符串。例如: “10 &lt;num &lt;= 30 and text != '这里什么都没有' 和 __dj__meta__.a = = 3”。针对每个样本评估条件，并且仅保留满足条件的样本。过滤条件的结果存储在密钥 “general_field_filter_condition” 下的样本统计信息中。如果筛选条件为空或已计算，则不重新评估样本。

Type 算子类型: **filter**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `filter_condition` | <class 'str'> | `''` | The filter condition as a string. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/general_field_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_general_field_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)