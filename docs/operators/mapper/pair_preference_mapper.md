# pair_preference_mapper

Mapper to construct paired preference samples by generating a rejected response and its
reason.

This operator uses an API model to generate a new response that is opposite in style,
factuality, or stance to the original response. The generated response and the reason
for its generation are stored in the sample. The default system prompt and input
template are provided, but can be customized. The output is parsed using a regular
expression to extract the new response and the reason. If parsing fails, the operator
retries up to a specified number of times. The generated response and reason are stored
in the sample under the keys 'rejected_response' and 'reason', respectively.

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the generation task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. It must |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `rejected_key` | <class 'str'> | `'rejected_response'` | The field name in the sample to store the |
| `reason_key` | <class 'str'> | `'reason'` | The field name in the sample to store the reason for |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retries for the API call in case of |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/pair_preference_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_pair_preference_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)