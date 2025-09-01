# relation_identity_mapper

Identify the relation between two entities in a given text.

This operator uses an API model to analyze the relationship between two specified entities in the text. It constructs a prompt with the provided system and input templates, then sends it to the API model for analysis. The output is parsed using a regular expression to extract the relationship. If the two entities are the same, the relationship is identified as "another identity." The result is stored in the meta field under the key 'role_relation' by default. The operator retries the API call up to a specified number of times in case of errors. If `drop_text` is set to True, the original text is removed from the sample after processing.

确定给定文本中两个实体之间的关系。

此运算符使用API模型来分析文本中两个指定实体之间的关系。它使用提供的系统和输入模板构建提示，然后将其发送到API模型进行分析。使用正则表达式解析输出以提取关系。如果两个实体相同，则关系被标识为 “另一身份”。默认情况下，结果存储在键 “role_relation” 下的元字段中。如果出现错误，操作员将重试API调用指定的次数。如果 “drop_text” 设置为True，则处理后将从样本中删除原始文本。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `source_entity` | <class 'str'> | `None` | The source entity of the relation to be |
| `target_entity` | <class 'str'> | `None` | The target entity of the relation to be |
| `output_key` | <class 'str'> | `'role_relation'` | The output key in the meta field in the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt_template` | typing.Optional[str] | `None` | System prompt template for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `output_pattern_template` | typing.Optional[str] | `None` | Regular expression template for |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `drop_text` | <class 'bool'> | `False` | If drop the text in the output. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/relation_identity_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_relation_identity_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)