# python_lambda_mapper

Mapper for applying a Python lambda function to data samples.

This operator allows users to define a custom transformation using a Python lambda function. The lambda function is applied to each sample, and the result must be a dictionary. If the `batched` parameter is set to True, the lambda function will process a batch of samples at once. If no lambda function is provided, the identity function is used, which returns the input sample unchanged. The operator validates the lambda function to ensure it has exactly one argument and compiles it safely.

Mapper，用于将Python lambda函数应用于数据样本。

此运算符允许用户使用Python lambda函数定义自定义转换。lambda函数应用于每个示例，结果必须是字典。如果 “batched” 参数设置为True，则lambda函数将立即处理一批样本。如果未提供lambda函数，则使用identity函数，该函数将返回未更改的输入样本。该运算符验证lambda函数以确保它只有一个参数并安全地编译它。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `lambda_str` | <class 'str'> | `''` | A string representation of the lambda function to be |
| `batched` | <class 'bool'> | `False` | A boolean indicating whether to process input data in |
| `kwargs` |  | `''` | Additional keyword arguments passed to the parent class. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/python_lambda_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_python_lambda_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)