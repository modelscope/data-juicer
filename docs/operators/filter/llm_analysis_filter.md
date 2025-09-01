# llm_analysis_filter

Base filter class for leveraging LLMs to analyze and filter data samples.

This operator uses a Hugging Face or API-based LLM to score and tag data samples across multiple dimensions. It evaluates each sample on clarity, relevance, usefulness, and fluency, providing scores from 1 to 5. The operator also assigns descriptive tags and flags for further review. The average score is computed based on the specified required keys, and samples with an average score below the minimum threshold are filtered out. The analysis results, including scores, tags, and flags, are stored in the sample's stats field under 'llm_analysis_score' and 'llm_analysis_record'. Samples are kept if their average score meets or exceeds the minimum threshold; otherwise, they are discarded.

用于利用LLMs分析和过滤数据样本的基本筛选器类。

此运算符使用拥抱面或基于API的LLM对跨多个维度的数据样本进行评分和标记。它评估每个样本的清晰度，相关性，有用性和流畅性，提供1到5的分数。操作员还分配描述性标签和标志以供进一步检查。根据指定的所需键计算平均得分，并过滤出平均得分低于最小阈值的样本。分析结果 (包括分数、标记和标志) 存储在 “llm_analysis_score” 和 “llm_analysis_record” 下的样本统计字段中。如果样本的平均得分满足或超过最小阈值，则保留样本; 否则，丢弃它们。

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
- [source code 源代码](../../../data_juicer/ops/filter/llm_analysis_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_analysis_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)