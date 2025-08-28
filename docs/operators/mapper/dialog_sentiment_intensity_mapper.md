# dialog_sentiment_intensity_mapper

Mapper to predict user's sentiment intensity in a dialog, ranging from -5 to 5.

This operator analyzes the sentiment of user queries in a dialog and outputs a list of
sentiment intensities and corresponding analyses. The sentiment intensity ranges from -5
(extremely negative) to 5 (extremely positive), with 0 indicating a neutral sentiment.
The analysis is based on the provided history, query, and response keys. The default
system prompt and templates guide the sentiment analysis process. The results are stored
in the meta field under 'dialog_sentiment_intensity' for intensities and
'dialog_sentiment_intensity_analysis' for analyses. The operator uses an API model to
generate the sentiment analysis, with configurable retry attempts and sampling
parameters.

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `max_round` | typing.Annotated[int, Ge(ge=0)] | `10` | The max num of round in the dialog to build the |
| `intensities_key` | <class 'str'> | `'dialog_sentiment_intensity'` | The key name in the meta field to store |
| `analysis_key` | <class 'str'> | `'dialog_sentiment_intensity_analysis'` | The key name in the meta field to store the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `query_template` | typing.Optional[str] | `None` | Template for query part to build the input |
| `response_template` | typing.Optional[str] | `None` | Template for response part to build the |
| `analysis_template` | typing.Optional[str] | `None` | Template for analysis part to build the |
| `intensity_template` | typing.Optional[str] | `None` | Template for intensity part to build the |
| `analysis_pattern` | typing.Optional[str] | `None` | Pattern to parse the return sentiment |
| `intensity_pattern` | typing.Optional[str] | `None` | Pattern to parse the return sentiment |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_sentiment_intensity_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_sentiment_intensity_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)