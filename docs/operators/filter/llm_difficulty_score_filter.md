# llm_difficulty_score_filter

Filter to keep samples with a high difficulty score estimated by an LLM.

This operator evaluates the difficulty of each sample using a large language model (LLM) and retains only those with a difficulty score above a specified threshold. The LLM analyzes the sample across multiple dimensions, including linguistic complexity, conceptual depth, prior knowledge, step complexity, and ambiguity. Each dimension is scored on a 1-5 scale, where 1 is novice-friendly and 5 is expert-level. The overall difficulty score is computed as the average of these dimension scores. The operator uses a Hugging Face tokenizer for text processing. The difficulty score is cached in the 'llm_difficulty_score' field, and detailed analysis is stored in 'llm_difficulty_record'.

过滤器，以保留由LLM估计的高难度分数的样本。

该运算符使用大型语言模型 (LLM) 评估每个样本的难度，并仅保留难度得分高于指定阈值的样本。LLM跨多个维度分析样本，包括语言复杂性，概念深度，先验知识，步骤复杂性和歧义。每个维度的评分为1-5，其中1为新手友好，5为专家级。总体难度分数被计算为这些维度分数的平均值。操作员使用拥抱面标记器进行文本处理。难度分数缓存在 “llm_difficulty_score” 字段中，详细分析存储在 “llm_difficulty_record” 中。

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
- [source code 源代码](../../../data_juicer/ops/filter/llm_difficulty_score_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_difficulty_score_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)