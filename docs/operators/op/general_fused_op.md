# general_fused_op

An explicitly fused operator designed to execute multiple sequential operations (OPs) on
the same batch, enabling fine-grained control over data processing.

This operator allows for the chaining of multiple data processing steps, such as mappers
and filters, into a single pass. It processes each batch of samples sequentially through
the defined operations, ensuring that all specified transformations are applied in
order. The operator supports both mappers, which transform data, and filters, which
remove or keep samples based on computed statistics. Context variables can be passed
between operations if needed. The accelerator is set to 'cuda' if any of the fused
operations use it. The number of processes is determined by the minimum value among all
fused operations. After processing, any temporary context variables, such as those used
for video containers, are cleaned up.

Type 算子类型: **op**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `batch_size` | <class 'int'> | `1` |  |
| `fused_op_list` | typing.List | `None` |  |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/op/general_fused_op.py)
- [unit test 单元测试]()
- [Return operator list 返回算子列表](../../Operators.md)