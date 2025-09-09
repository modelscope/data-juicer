# image_pair_similarity_filter

Filter to keep image pairs with similarities between images within a specific range.

This operator uses a Hugging Face CLIP model to compute the cosine similarity between two images in each sample. It retains samples where the similarity score falls within the specified minimum and maximum thresholds. The 'any' strategy keeps a sample if any of the image pairs meet the condition, while the 'all' strategy requires all image pairs to meet the condition. The similarity scores are cached in the 'image_pair_similarity' field. Each sample must include exactly two distinct images.

用于保留图像之间相似度在特定范围内的图像对的过滤器。

该算子使用Hugging Face CLIP模型来计算每个样本中两张图像之间的余弦相似度。如果相似度得分落在指定的最小和最大阈值范围内，则保留样本。'any'策略要求至少有一对图像满足条件即可保留样本，而'all'策略要求所有图像对都必须满足条件才能保留样本。相似度得分缓存在'image_pair_similarity'字段中。每个样本必须包含两张不同的图像。

Type 算子类型: **filter**

Tags 标签: cpu, hf, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_clip` |  | `'openai/clip-vit-base-patch32'` | clip model name on huggingface to compute |
| `trust_remote_code` |  | `False` |  |
| `min_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `0.1` | The min similarity to keep samples. |
| `max_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `1.0` | The max similarity to keep samples. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_no_eoc_special_token
```python
ImagePairSimilarityFilter(hf_clip='openai/clip-vit-base-patch32', any_or_all='any', min_score=0.85, max_score=1)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">image pair 1</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">image pair 2</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img3.jpg|img7.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img7.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">image pair 3</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg|img5.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img5.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">image pair 2</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img3.jpg|img7.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img7.jpg" width="160" style="margin:4px;"/></div></div><div class='meta' style='margin-top:6px;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #eaecef !important;'><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; background-color:#f8f9fa !important; border-bottom:1px solid #eaecef !important; font-weight:bold; color:#555;'>__dj__stats__</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; background-color:#f8f9fa !important; border-bottom:1px solid #eaecef !important; white-space:nowrap; padding-left: 16px;'>image_pair_similarity</td><td style='text-align:left; vertical-align:top; padding:4px 8px; background-color:#f8f9fa !important; border-bottom:1px solid #eaecef !important;'>[0.9999999403953552]</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to filter out image pairs based on their similarity scores. The operator uses a Hugging Face CLIP model to calculate the cosine similarity between two images in each sample. In this case, the operator is set to keep samples where any of the image pairs have a similarity score between 0.85 and 1. The output data shows that only 'image pair 2' is retained because its images (img3.jpg and img7.jpg) have a similarity score of approximately 1, which falls within the specified range. The 'meta' field in the output contains the calculated similarity score for the retained sample.
这个例子展示了算子基于图像对之间的相似度分数进行过滤的能力。算子使用Hugging Face的CLIP模型来计算每个样本中两个图像之间的余弦相似度。在这种情况下，算子设置为保留任何图像对的相似度分数在0.85到1之间的样本。输出数据显示只有'image pair 2'被保留了，因为它的图像（img3.jpg 和 img7.jpg）的相似度分数约为1，落在指定范围内。输出中的'meta'字段包含了保留样本的计算出的相似度分数。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_pair_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_pair_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)