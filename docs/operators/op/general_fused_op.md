# general_fused_op

An explicitly fused operator designed to execute multiple sequential operations (OPs) on the same batch, enabling fine-grained control over data processing.

This operator allows for the chaining of multiple data processing steps, such as mappers and filters, into a single pass. It processes each batch of samples sequentially through the defined operations, ensuring that all specified transformations are applied in order. The operator supports both mappers, which transform data, and filters, which remove or keep samples based on computed statistics. Context variables can be passed between operations if needed. The accelerator is set to 'cuda' if any of the fused operations use it. The number of processes is determined by the minimum value among all fused operations. After processing, any temporary context variables, such as those used for video containers, are cleaned up.

一种显式融合运算符，旨在在同一批上执行多个顺序操作 (OPs)，从而实现对数据处理的细粒度控制。

该操作符允许将多个数据处理步骤 (例如映射器和过滤器) 链接到单个通道中。它通过定义的操作顺序处理每批样品，确保按顺序应用所有指定的转换。该运算符支持转换数据的映射器和基于计算统计信息删除或保留样本的过滤器。如果需要，可以在操作之间传递上下文变量。如果任何融合的操作使用它，则加速器被设置为 “cuda'”。处理的数量由所有融合操作中的最小值确定。在处理之后，任何临时上下文变量 (诸如用于视频容器的那些) 被清除。

Type 算子类型: **op**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `batch_size` | <class 'int'> | `1` | the batch size of the input samples. |
| `fused_op_list` | typing.Optional[typing.List] | `None` | a list of OPs to be fused. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/op/general_fused_op.py)
- [unit test 单元测试]()
- [Return operator list 返回算子列表](../../Operators.md)