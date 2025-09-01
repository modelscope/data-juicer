# entity_attribute_aggregator

Summarizes a given attribute of an entity from a set of documents.

- The operator extracts and summarizes the specified attribute of a given entity from the provided documents.
- It uses a system prompt, example prompt, and input template to generate the summary.
- The output is formatted as a markdown-style summary with the entity and attribute clearly labeled.
- The summary is limited to a specified number of words (default is 100).
- The operator uses a Hugging Face tokenizer to handle token limits and splits documents if necessary.
- If the input key or required fields are missing, the operator logs a warning and returns the sample unchanged.
- The summary is stored in the batch metadata under the specified output key.

汇总一组文档中实体的给定属性。

- 运算符从提供的文档中提取并汇总给定实体的指定属性。
- 它使用系统提示，示例提示和输入模板来生成摘要。
- 输出的格式为markdown样式的摘要，带有明确标记的实体和属性。
- 摘要限制为指定数量的单词 (默认为100)。
- 操作员使用拥抱面标记器来处理标记限制，并在必要时拆分文档。
- 如果缺少输入键或必填字段，则操作员会记录警告并返回未更改的示例。
- 摘要存储在指定输出键下的批处理元数据中。

Type 算子类型: **aggregator**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `entity` | <class 'str'> | `None` | The given entity. |
| `attribute` | <class 'str'> | `None` | The given attribute. |
| `input_key` | <class 'str'> | `'event_description'` | The input key in the meta field of the samples. |
| `output_key` | <class 'str'> | `'entity_attribute'` | The output key in the aggregation field of the |
| `word_limit` | typing.Annotated[int, Gt(gt=0)] | `100` | Prompt the output length. |
| `max_token_num` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The max token num of the total tokens of the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt_template` | typing.Optional[str] | `None` | The system prompt template. |
| `example_prompt` | typing.Optional[str] | `None` | The example part in the system prompt. |
| `input_template` | typing.Optional[str] | `None` | The input template. |
| `output_pattern_template` | typing.Optional[str] | `None` | The output template. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/aggregator/entity_attribute_aggregator.py)
- [unit test 单元测试](../../../tests/ops/aggregator/test_entity_attribute_aggregator.py)
- [Return operator list 返回算子列表](../../Operators.md)