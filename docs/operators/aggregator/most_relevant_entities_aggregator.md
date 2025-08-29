# most_relevant_entities_aggregator

Extracts and ranks entities closely related to a given entity from provided texts.
- The operator uses a language model API to identify and rank entities.
- It filters out entities of the same type as the given entity.
- The ranked list is sorted in descending order of importance.
- The input texts are aggregated and passed to the model, with a token limit if
specified.
- The output is parsed using a regular expression to extract the relevant entities.
- Results are stored in the batch metadata under the key 'most_relevant_entities'.
- The operator retries the API call up to a specified number of times in case of errors.

Type 算子类型: **aggregator**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `entity` | <class 'str'> | `None` | The given entity. |
| `query_entity_type` | <class 'str'> | `None` | The type of queried relevant entities. |
| `input_key` | <class 'str'> | `'event_description'` | The input key in the meta field of the samples. |
| `output_key` | <class 'str'> | `'most_relevant_entities'` | The output key in the aggregation field of the |
| `max_token_num` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The max token num of the total tokens of the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt_template` | typing.Optional[str] | `None` | The system prompt template. |
| `input_template` | typing.Optional[str] | `None` | The input template. |
| `output_pattern` | typing.Optional[str] | `None` | The output pattern. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/aggregator/most_relevant_entities_aggregator.py)
- [unit test 单元测试](../../../tests/ops/aggregator/test_most_relevant_entities_aggregator.py)
- [Return operator list 返回算子列表](../../Operators.md)