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
### test_api
```python
TextEmbdSimilarityFilter(api_or_hf_model='text-embedding-v4', is_hf_model=False, min_score=0.7, max_score=1.0, ebd_dim=2048)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">There is a lovely cat.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">It is challenging to train a large language model.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">There is a lovely cat.</pre></div>

#### ✨ explanation 解释
The operator filters texts based on their average embedding similarity to a set of validation texts. It keeps the text 'There is a lovely cat.' because its similarity score falls within the specified range [0.7, 1.0], while the other text is removed due to a lower similarity score.
算子根据文本与一组验证文本的平均嵌入相似度来过滤文本。它保留了文本'There is a lovely cat.'，因为其相似度得分落在指定范围[0.7, 1.0]内，而另一个文本由于相似度得分较低被移除。

### test_rft_data
```python
TextEmbdSimilarityFilter(api_or_hf_model='text-embedding-v4', is_hf_model=False, min_score=0.2, max_score=1.0, ebd_dim=2048, input_template='{text} {analysis} {answer}')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">It is challenging to train a large language model.</pre><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>analysis</td><td style='padding:4px 8px;'></td></tr></table></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> What is the capital of France?</div><div><strong>A:</strong> The capital of France is Paris.</div></div><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>analysis</td><td style='padding:4px 8px;'>The question asks for a factual piece of information about the capital city of France. The answer is straightforward and...</td></tr></table></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?</div><div><strong>A:</strong> 624</div></div><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>analysis</td><td style='padding:4px 8px;'>He writes each friend 3*2=&lt;&lt;3*2=6&gt;&gt;6 pages a week So he writes 6*2=&lt;&lt;6*2=12&gt;&gt;12 pages every week That means he writes 12...</td></tr></table></details></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to handle more complex input structures, including analysis and answer fields. The operator computes the similarity using an input template that combines these fields. Only the text about James writing letters is kept as it has a similarity score within the range [0.2, 1.0].
这个例子展示了算子处理更复杂输入结构的能力，包括分析和答案字段。算子使用一个将这些字段组合起来的输入模板来计算相似度。只有关于James写信的文本被保留下来，因为它在[0.2, 1.0]范围内的相似度得分。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/text_embd_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_text_embd_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)