# dialog_topic_detection_mapper

Generates user's topic labels and analysis in a dialog.

This operator processes a dialog to detect and label the topics discussed by the user.
It takes input from `history_key`, `query_key`, and `response_key` and outputs lists of
labels and analysis for each query in the dialog. The operator uses a predefined system
prompt and templates to build the input prompt for the API call. It supports customizing
the system prompt, templates, and patterns for parsing the API response. The results are
stored in the `meta` field under the keys specified by `labels_key` and `analysis_key`.
If these keys already exist in the `meta` field, the operator skips processing. The
operator retries the API call up to `try_num` times in case of errors.

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `topic_candidates` | typing.Optional[typing.List[str]] | `None` | The output topic candidates. Use |
| `max_round` | typing.Annotated[int, Ge(ge=0)] | `10` | The max num of round in the dialog to build the |
| `labels_key` | <class 'str'> | `'dialog_topic_labels'` | The key name in the meta field to store the |
| `analysis_key` | <class 'str'> | `'dialog_topic_labels_analysis'` | The key name in the meta field to store the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `query_template` | typing.Optional[str] | `None` | Template for query part to build the input |
| `response_template` | typing.Optional[str] | `None` | Template for response part to build the |
| `candidate_template` | typing.Optional[str] | `None` | Template for topic candidates to |
| `analysis_template` | typing.Optional[str] | `None` | Template for analysis part to build the |
| `labels_template` | typing.Optional[str] | `None` | Template for labels part to build the |
| `analysis_pattern` | typing.Optional[str] | `None` | Pattern to parse the return topic |
| `labels_pattern` | typing.Optional[str] | `None` | Pattern to parse the return topic |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_topic_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_topic_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)