# extract_support_text_mapper

Extracts a supporting sub-text from the original text based on a given summary.

This operator uses an API model to identify and extract a segment of the original text
that best matches the provided summary. It leverages a system prompt and input template
to guide the extraction process. The extracted support text is stored in the specified
meta field key. If the extraction fails or returns an empty string, the original summary
is used as a fallback. The operator retries the extraction up to a specified number of
times in case of errors.

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `summary_key` | <class 'str'> | `'event_description'` | The key name to store the input summary in the |
| `support_text_key` | <class 'str'> | `'support_text'` | The key name to store the output |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_support_text_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_support_text_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)