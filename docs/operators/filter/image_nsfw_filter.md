# image_nsfw_filter

Filter to keep samples whose images have low NSFW scores.

This operator uses a Hugging Face model to detect and score the NSFW content in images.
It keeps samples if their image NSFW scores are below a specified threshold. The
filtering strategy can be set to 'any' (keep if any image meets the condition) or 'all'
(keep only if all images meet the condition). The key metric is the NSFW score, computed
for each image and stored in the 'image_nsfw_score' field. If no images are present, an
empty array is stored.

Type 算子类型: **filter**

Tags 标签: cpu, hf, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_nsfw_model` | <class 'str'> | `'Falconsai/nsfw_image_detection'` | nsfw detection model name on huggingface. |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `min_score` | <class 'float'> | `0.0` | the min nsfw score threshold for samples. |
| `max_score` | <class 'float'> | `0.5` | the max nsfw score threshold for samples. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_nsfw_filter
```python
ImageNSFWFilter(hf_nsfw_model=self.hf_nsfw_model, max_score=0.0005)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator filters out samples with images having NSFW scores above 0.0005, keeping only those below this threshold. The first sample is removed because its image has a higher NSFW score than the set maximum.
算子过滤掉NSFW分数高于0.0005的图像样本，只保留低于此阈值的样本。第一个样本被移除是因为它的图像NSFW分数超过了设定的最大值。

### test_all
```python
ImageNSFWFilter(hf_nsfw_model=self.hf_nsfw_model, max_score=0.0005, any_or_all='all')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The 'all' strategy requires all images in a sample to have NSFW scores below 0.0005 for the sample to be kept. The first sample is removed since not all of its images meet this condition, while the second one is retained as both images have acceptable NSFW scores.
'all'策略要求样本中的所有图像都必须具有低于0.0005的NSFW分数才能保留该样本。第一个样本被移除，因为并非其所有图像都满足这个条件；而第二个样本被保留下来，因为它的两个图像都有可接受的NSFW分数。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_nsfw_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_nsfw_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)