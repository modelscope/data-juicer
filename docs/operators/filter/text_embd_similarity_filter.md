# text_embd_similarity_filter

Filter to keep texts whose average embedding similarity to a set of given validation
texts falls within a specific range.

This operator computes the cosine similarity between the text embeddings and a set of
validation text embeddings. It keeps samples where the average similarity score is
within the specified range. The key metric, 'text_embd_similarity', is computed as the
mean cosine similarity. The operator supports both API-based and Hugging Face model-
based embeddings. If no valid dataset is provided, the `prepare_valid_feature` method
must be called manually before applying the filter.

Type 算子类型: **filter**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'text-embedding-v4'` | API or huggingface embedding model name. |
| `is_hf_model` | <class 'bool'> | `False` | Indicates if the model is from HuggingFace. |
| `api_endpoint` | <class 'str'> | `'embeddings'` | Embedding URL endpoint for the API. |
| `response_path` | <class 'str'> | `'data.0.embedding'` | Path to extract content from the API response. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the API model. |
| `min_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `0.1` | The min average similarity to keep samples. |
| `max_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `1.0` | The max average similarity to keep samples. |
| `valid_dataset` | typing.Optional[typing.List[typing.Dict]] | `None` | The dataset to use for validation. |
| `ebd_dim` | <class 'int'> | `4096` | The embedding's dimension via API. |
| `pooling` | typing.Optional[str] | `None` | strategy to extract embedding from the hidden states. https://arxiv.org/abs/2503.01807 |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/text_embd_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_text_embd_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)