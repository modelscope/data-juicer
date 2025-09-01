# python_file_mapper

Executes a Python function defined in a file on input data.

This operator loads a specified Python function from a given file and applies it to the input data. The function must take exactly one argument and return a dictionary. The operator can process data either sample by sample or in batches, depending on the `batched` parameter. If the file path is not provided, the operator acts as an identity function, returning the input sample unchanged. The function is loaded dynamically, and its name and file path are configurable. Important notes:
- The file must be a valid Python file (`.py`).
- The function must be callable and accept exactly one argument.
- The function's return value must be a dictionary.

对输入数据执行文件中定义的Python函数。

此运算符从给定文件加载指定的Python函数，并将其应用于输入数据。该函数必须只接受一个参数并返回一个字典。操作员可以根据 “批量” 参数逐样本或批量处理数据。如果未提供文件路径，则运算符充当标识函数，返回不变的输入样本。该函数是动态加载的，其名称和文件路径是可配置的。重要注意事项:
- 文件必须是有效的Python文件 ('.py')。
- 函数必须是可调用的，并且只接受一个参数。
- 函数的返回值必须是字典。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `file_path` | <class 'str'> | `''` | The path to the Python file containing the function |
| `function_name` | <class 'str'> | `'process_single'` | The name of the function defined in the file |
| `batched` | <class 'bool'> | `False` | A boolean indicating whether to process input data in |
| `kwargs` |  | `''` | Additional keyword arguments passed to the parent class. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/python_file_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_python_file_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)