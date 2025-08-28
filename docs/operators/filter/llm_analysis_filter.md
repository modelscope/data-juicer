# llm_analysis_filter

Base filter class for leveraging LLMs to analyze and filter data samples.

This operator uses a Hugging Face or API-based LLM to score and tag data samples across
multiple dimensions. It evaluates each sample on clarity, relevance, usefulness, and
fluency, providing scores from 1 to 5. The operator also assigns descriptive tags and
flags for further review. The average score is computed based on the specified required
keys, and samples with an average score below the minimum threshold are filtered out.
The analysis results, including scores, tags, and flags, are stored in the sample's
stats field under 'llm_analysis_score' and 'llm_analysis_record'. Samples are kept if
their average score meets or exceeds the minimum threshold; otherwise, they are
discarded.

Type 算子类型: **filter**

Tags 标签: cpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'gpt-4o'` | API or huggingface model name. |
| `min_score` | <class 'float'> | `0.5` | The lowest score threshold to keep |
| `is_hf_model` | <class 'bool'> | `False` |  |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `input_keys` | typing.List[str] | `['text']` | Sub set of keys in the sample. Support data with |
| `field_names` | typing.List[str] | `['Text']` | Corresponding field names for input keys. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `field_template` | typing.Optional[str] | `None` | Template for each field in the prompt. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `enable_vllm` | <class 'bool'> | `False` | If true, use VLLM for loading hugging face or |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `dim_required_keys` | typing.Optional[typing.List[str]] | `None` | A list of keys used to calculate the average |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/llm_analysis_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_analysis_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)