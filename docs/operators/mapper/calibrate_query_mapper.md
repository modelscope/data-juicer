# calibrate_query_mapper

Calibrate query in question-answer pairs based on reference text.

This operator adjusts the query (question) in a question-answer pair to be more detailed
and accurate, while ensuring it can still be answered by the original answer. It uses a
reference text to inform the calibration process. The calibration is guided by a system
prompt, which instructs the model to refine the question without adding extraneous
information. The output is parsed to extract the calibrated query, with any additional
content removed.

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the calibration task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `reference_template` | typing.Optional[str] | `None` | Template for formatting the reference text. |
| `qa_pair_template` | typing.Optional[str] | `None` | Template for formatting question-answer pairs. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/calibrate_query_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_calibrate_query_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)