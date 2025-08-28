# extract_nickname_mapper

Extracts nickname relationships in the text using a language model.

This operator uses a language model to identify and extract nickname relationships from
the input text. It follows specific instructions to ensure accurate extraction, such as
identifying the speaker, the person being addressed, and the nickname used. The
extracted relationships are stored in the meta field under the specified key. The
operator uses a default system prompt, input template, and output pattern, but these can
be customized. The results are parsed and validated to ensure they meet the required
format. If the text already contains the nickname information, it is not processed
again. The operator retries the API call a specified number of times if an error occurs.

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `nickname_key` | <class 'str'> | `'nickname'` | The key name to store the nickname |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_nickname_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_nickname_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)