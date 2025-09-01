# naive_grouper

Group all samples in a dataset into a single batched sample.

This operator takes a dataset and combines all its samples into one batched sample. If the input dataset is empty, it returns an empty dataset. The resulting batched sample is a dictionary where each key corresponds to a list of values from all samples in the dataset.

将数据集中的所有样本分组为单个批处理样本。

此运算符获取一个数据集，并将其所有样本合并为一个批处理样本。如果输入数据集为空，则返回空数据集。生成的批处理样本是一个字典，其中每个键对应于数据集中的所有样本的值列表。

Type 算子类型: **grouper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/grouper/naive_grouper.py)
- [unit test 单元测试](../../../tests/ops/grouper/test_naive_grouper.py)
- [Return operator list 返回算子列表](../../Operators.md)