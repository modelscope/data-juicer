# python_file_mapper

Executes a Python function defined in a file on input data.

This operator loads a specified Python function from a given file and applies it to the
input data. The function must take exactly one argument and return a dictionary. The
operator can process data either sample by sample or in batches, depending on the
`batched` parameter. If the file path is not provided, the operator acts as an identity
function, returning the input sample unchanged. The function is loaded dynamically, and
its name and file path are configurable. Important notes:
- The file must be a valid Python file (`.py`).
- The function must be callable and accept exactly one argument.
- The function's return value must be a dictionary.

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