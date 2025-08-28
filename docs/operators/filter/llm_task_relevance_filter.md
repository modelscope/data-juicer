# llm_task_relevance_filter

Filter to keep samples with high relevance scores to validation tasks estimated by an
LLM.

This operator evaluates the relevance of each sample to a specified validation task
using an LLM. The LLM scores the sample on multiple dimensions, including topical
relevance, linguistic style match, task match, knowledge alignment, and potential
utility. Each dimension is scored on a 1-5 scale, with 5 being the highest. The key
metric, 'llm_task_relevance', is the average score across these dimensions. Samples are
kept if their average score meets or exceeds the specified minimum threshold. The
operator uses either an API or a Hugging Face model for evaluation. If no validation
dataset or task description is provided, the 'prepare_valid_feature' method must be
called manually before applying the filter.

Type 算子类型: **filter**

Tags 标签: cpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'gpt-4o'` | API or huggingface model name. |
| `min_score` | <class 'float'> | `0.5` | The lowest score threshold to keep the sample. |
| `is_hf_model` | <class 'bool'> | `False` | Indicates if the model is from HuggingFace. |
| `valid_dataset` | typing.Optional[typing.List[typing.Dict]] | `None` | The dataset to use for validation. |
| `task_desc` | typing.Optional[str] | `None` | The description of the validation task. |
| `n_shot` | typing.Optional[int] | `None` | The number of shots in validation. |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/llm_task_relevance_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_task_relevance_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)