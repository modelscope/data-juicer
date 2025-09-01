# nested_aggregator

Aggregates nested content from multiple samples into a single summary.

This operator uses a recursive summarization approach to aggregate content from multiple samples. It processes the input text, which is split into sub-documents, and generates a summary that maintains the average length of the original documents. The aggregation is performed using an API model, and the process is guided by system prompts and templates. The operator supports retrying the API call in case of errors and allows for customization of the summarization process through various parameters. The default system prompt and templates are provided in Chinese, and the final summary is expected to be in the same language.

将多个示例中的嵌套内容聚合到单个摘要中。

此运算符使用递归汇总方法来聚合来自多个样本的内容。它处理被分割成子文档的输入文本，并生成保持原始文档的平均长度的摘要。使用API模型执行聚合，并且该过程由系统提示和模板指导。操作员支持在错误的情况下重试API调用，并允许通过各种参数定制汇总过程。默认的系统提示和模板以中文提供，最终摘要预计使用相同的语言。

Type 算子类型: **aggregator**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `input_key` | <class 'str'> | `'event_description'` | The input key in the meta field of the samples. |
| `output_key` | <class 'str'> | `None` | The output key in the aggregation field in the |
| `max_token_num` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The max token num of the total tokens of the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | The system prompt. |
| `sub_doc_template` | typing.Optional[str] | `None` | The template for input text in each sample. |
| `input_template` | typing.Optional[str] | `None` | The input template. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/aggregator/nested_aggregator.py)
- [unit test 单元测试](../../../tests/ops/aggregator/test_nested_aggregator.py)
- [Return operator list 返回算子列表](../../Operators.md)