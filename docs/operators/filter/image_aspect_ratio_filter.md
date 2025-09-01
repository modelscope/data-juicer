# image_aspect_ratio_filter

Filter to keep samples with image aspect ratio within a specific range.

The operator computes the aspect ratio for each image in the sample, defined as the width divided by the height (W / H). It caches the computed aspect ratios in the 'aspect_ratios' field. Samples are kept if their images' aspect ratios fall within the specified minimum and maximum range. The 'any_or_all' parameter determines the strategy: 'any' keeps samples if at least one image meets the criteria, while 'all' requires all images to meet the criteria. If no images are present in a sample, the sample is not filtered out.

过滤器，以保持样本的图像纵横比在特定范围内。

运算符计算样本中每个图像的纵横比，定义为宽度除以高度 (W/H)。它将计算的纵横比缓存在 'aspect_raties' 字段中。如果其图像的纵横比落在指定的最小和最大范围内，则保留样本。“any_or_all” 参数确定策略: 如果至少一个图像满足标准，则 “any” 保持采样，而 “all” 要求所有图像满足标准。如果在样本中不存在图像，则不滤出样本。

Type 算子类型: **filter**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_ratio` | <class 'float'> | `0.333` | The min aspect ratio to keep samples. |
| `max_ratio` | <class 'float'> | `3.0` | The max aspect ratio to keep samples. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_filter1
```python
ImageAspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator filters the samples to keep only those with an image aspect ratio between 0.8 and 1.2. Only img1 meets this criteria, so it is the only one kept in the target list.
算子过滤样本，只保留图像宽高比在0.8到1.2之间的样本。只有img1符合这个条件，所以目标列表中只有它被保留下来。

### test_any
```python
ImageAspectRatioFilter(min_ratio=0.8, max_ratio=1.2, any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator filters the samples to keep those where at least one image has an aspect ratio between 0.8 and 1.2. Both the first and third samples have at least one image (img1) that meets the criteria, so they are both kept in the target list.
算子过滤样本，只要至少有一张图片的宽高比在0.8到1.2之间就保留该样本。第一个和第三个样本中都至少有一张图片（img1）符合此条件，因此它们都被保留在目标列表中。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_aspect_ratio_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_aspect_ratio_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)