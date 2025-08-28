# image_text_similarity_filter

Filter to keep samples with image-text similarity within a specified range.

This operator uses a Hugging Face CLIP model to compute the similarity between images
and text. It retains samples where the similarity scores fall within the given range.
The similarity score is computed for each image-text pair, and the final score can be
reduced using 'avg', 'max', or 'min' modes. The 'any' or 'all' strategy determines if at
least one or all image-text pairs must meet the similarity criteria. The key metric
'image_text_similarity' is cached in the sample's stats. Images can be flipped
horizontally or vertically before computing the similarity.

Type 算子类型: **filter**

Tags 标签: cpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_clip` | <class 'str'> | `'openai/clip-vit-base-patch32'` | clip model name on huggingface to compute |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `min_score` | <class 'float'> | `0.1` | The min similarity to keep samples. |
| `max_score` | <class 'float'> | `1.0` | The max similarity to keep samples. |
| `horizontal_flip` | <class 'bool'> | `False` | Flip image horizontally (left to right). |
| `vertical_flip` | <class 'bool'> | `False` | Flip image vertically (top to bottom). |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `reduce_mode` | <class 'str'> | `'avg'` | reduce mode when one text corresponds to |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_keep_any
```python
ImageTextSimilarityFilter(hf_clip=self.hf_clip, reduce_mode='avg', any_or_all='any', horizontal_flip=False, vertical_flip=False, min_score=0.2, max_score=0.9)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__image&gt;a photo of a cat &lt;|__dj__eoc|&gt; &lt;__dj__image&gt;a photo of a dog &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|cat.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__image&gt;a photo of a cat &lt;|__dj__eoc|&gt; &lt;__dj__image&gt;a photo of a dog &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|cat.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator retains samples where at least one image-text pair meets the similarity criteria. In this case, both pairs meet the criteria, so the sample is kept.
算子保留至少一个图像-文本对满足相似度标准的样本。在这种情况下，两个对都满足标准，因此样本被保留。

### test_reduce_min
```python
ImageTextSimilarityFilter(hf_clip=self.hf_clip, reduce_mode='min', any_or_all='any', horizontal_flip=False, vertical_flip=False, min_score=0.1, max_score=0.9)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__image&gt;a photo of a cat &lt;__dj__image&gt; &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__image&gt;a photo of a cat &lt;__dj__image&gt; &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator uses the 'min' reduce mode to compute the minimum similarity score among all image-text pairs in a sample. If the minimum score is within the specified range, the sample is kept; otherwise, it is removed. In this test, the sample is initially kept because the minimum score is 0.1, which is within the range. However, when the min_score is set to 0.2, the sample is removed as the minimum score no longer meets the new threshold.
算子使用'min'减少模式来计算样本中所有图像-文本对之间的最小相似度分数。如果最小分数在指定范围内，则保留该样本；否则，将其移除。在此测试中，样本最初被保留，因为最小分数为0.1，在范围内。但是，当将min_score设置为0.2时，由于最小分数不再达到新的阈值，样本被移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_text_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_text_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)