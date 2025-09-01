# llm_quality_score_filter

Filter to keep samples with a high quality score estimated by an LLM.

This operator uses a Hugging Face LLM to evaluate each sample across multiple quality dimensions, including accuracy, grammar, informativeness, and coherence. The LLM provides a numerical score for each dimension on a 1-5 scale, along with a rationale and recommendation. The overall quality score is then used to filter samples. Samples are kept if their quality score meets or exceeds the specified minimum score. The key metric is 'llm_quality_score', which is computed based on the LLM's evaluation. The LLM also provides a detailed record of its analysis, which is stored in 'llm_quality_record'.

过滤器，以保持LLM估计的高质量分数的样本。

此运算符使用拥抱面LLM在多个质量维度上评估每个样本，包括准确性，语法，信息性和连贯性。LLM在1-5量表上提供每个维度的数字分数，以及基本原理和建议。然后使用总体质量分数来过滤样本。如果样品的质量分数达到或超过指定的最低分数，则保留样品。关键指标是 “llm_quality_score”，它是根据LLM的评估计算得出的。LLM还提供了其分析的详细记录，该记录存储在 “llm_quality_record” 中。

Type 算子类型: **filter**

Tags 标签: cpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'gpt-4o'` | API or huggingface model name. |
| `min_score` | <class 'float'> | `0.5` | The min score threshold to keep the sample. |
| `max_score` | <class 'float'> | `1.0` | The max score threshold to keep the sample. |
| `is_hf_model` | <class 'bool'> | `False` | If true, use huggingface model. Otherwise, use API. |
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
- [source code 源代码](../../../data_juicer/ops/filter/llm_quality_score_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_quality_score_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)