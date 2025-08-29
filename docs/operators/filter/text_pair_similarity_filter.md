# text_pair_similarity_filter

Filter to keep text pairs with similarities within a specific range.

This operator computes the similarity between two texts in a pair using a Hugging Face
CLIP model. It keeps samples where the similarity score falls within the specified min
and max thresholds. The key metric, 'text_pair_similarity', is computed as the cosine
similarity between the text embeddings. The operator supports two strategies for keeping
samples: 'any' (keep if any pair meets the condition) and 'all' (keep only if all pairs
meet the condition). If the second text key is not provided, the operator will raise an
error. The similarity scores are cached under the 'text_pair_similarity' field in the
sample's stats.

Type 算子类型: **filter**

Tags 标签: cpu, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_clip` |  | `'openai/clip-vit-base-patch32'` | clip model name on huggingface to compute |
| `trust_remote_code` |  | `False` |  |
| `min_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `0.1` | The min similarity to keep samples. |
| `max_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `1.0` | The max similarity to keep samples. |
| `text_key_second` |  | `None` | used to store the other sentence |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_no_eoc_special_token
```python
TextPairSimilarityFilter(hf_clip=self.hf_clip, any_or_all='any', min_score=0.85, max_score=0.99, text_key_second=self.text_key_second)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a lovely cat</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>target_text</td><td style='padding:4px 8px;'>a lovely cat</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a cute cat</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>target_text</td><td style='padding:4px 8px;'>a lovely cat</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a black dog</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>target_text</td><td style='padding:4px 8px;'>a lovely cat</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a cute cat</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>target_text</td><td style='padding:4px 8px;'>a lovely cat</td></tr></table></div></div>

#### ✨ explanation 解释
The operator filters the input data to keep only those text pairs with a similarity score between 0.85 and 0.99. It uses a Hugging Face CLIP model to calculate the cosine similarity between the embeddings of each text pair. The 'any' strategy is selected, meaning a sample is kept if any of its text pairs meet the condition. In this case, the second sample ('a cute cat' and 'a lovely cat') has a similarity within the specified range, so it is kept, while the other samples are removed because their similarities fall outside the defined thresholds.
算子过滤输入数据，只保留相似度得分在0.85到0.99之间的文本对。它使用Hugging Face的CLIP模型计算每对文本嵌入之间的余弦相似度。选择了'any'策略，意味着如果样本中的任何一对文本满足条件，则保留该样本。在这种情况下，第二个样本（'一只可爱的小猫'和'一只漂亮的小猫'）的相似度在指定范围内，因此被保留，而其他样本由于其相似度不在定义的阈值范围内而被移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/text_pair_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_text_pair_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)