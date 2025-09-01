# naive_reverse_grouper

Split batched samples into individual samples.

This operator processes a dataset by splitting each batched sample into individual samples. It also handles and optionally exports batch metadata.
- If a sample contains 'batch_meta', it is separated and can be exported to a specified path.
- The operator converts the remaining data from a dictionary of lists to a list of dictionaries, effectively unbatching the samples.
- If `batch_meta_export_path` is provided, the batch metadata is written to this file in JSON format, one entry per line.
- If no samples are present in the dataset, the original dataset is returned.

将批处理的样品分成单个样品。

此运算符通过将每个批处理样本拆分为单个样本来处理数据集。它还处理和 (可选) 导出批处理元数据。
- 如果示例包含 'batch_meta'，则将其分开并可以导出到指定路径。
- 运算符将剩余数据从列表字典转换为字典列表，从而有效地对样本进行批处理。
- 如果提供了 'batch_meta_export_path'，则以JSON格式将批处理元数据写入此文件，每行一个条目。
- 如果数据集中不存在任何样本，则返回原始数据集。

Type 算子类型: **grouper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `batch_meta_export_path` |  | `None` | the path to export the batch meta. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/grouper/naive_reverse_grouper.py)
- [unit test 单元测试](../../../tests/ops/grouper/test_naive_reverse_grouper.py)
- [Return operator list 返回算子列表](../../Operators.md)