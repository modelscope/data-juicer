# image_size_filter

Keep data samples whose image sizes are within a specific range.

This operator filters data samples based on the size of their images. It retains samples if the image sizes fall within the specified minimum and maximum size range. The `any_or_all` parameter determines the filtering strategy: 'any' keeps the sample if any image meets the size criteria, while 'all' keeps the sample only if all images meet the criteria. The key metric `image_sizes` is computed in bytes. If the image sizes are already cached in the `image_sizes` field, they are reused. Otherwise, the operator calculates the sizes from the image byte data or file paths.

保留图像大小在特定范围内的数据样本。

此运算符根据图像的大小对数据样本进行过滤。如果图像大小在指定的最小和最大大小范围内，它将保留样本。“any_or_all” 参数确定过滤策略: 如果任何图像满足大小标准，则 “any” 保留样本，而 “all” 仅在所有图像满足标准时保留样本。关键指标 “image_sizes” 以字节为单位计算。如果图像大小已经缓存在 'image_sizes' 字段中，则重用它们。否则，运算符根据图像字节数据或文件路径计算大小。

Type 算子类型: **filter**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_size` | <class 'str'> | `'0'` | The min image size to keep samples.  set to be "0" by |
| `max_size` | <class 'str'> | `'1TB'` | The max image size to keep samples.  set to be |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_min_max
```python
ImageSizeFilter(min_size='120kb', max_size='180KB')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator filters out images that are not within the 120KB to 180KB size range. Only the image with a size of 171KB is kept, while the others are removed because they do not meet the size criteria.
算子过滤掉不在120KB到180KB大小范围内的图片。只有大小为171KB的图片被保留，其他图片因为不符合尺寸标准而被移除。

### test_any
```python
ImageSizeFilter(min_size='120kb', max_size='180KB', any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator retains samples if any of their images fall within the 120KB to 180KB size range. The first and third samples have at least one image within this range, so they are kept. The second sample is removed as both its images are outside the specified size range.
如果样本中的任何一张图片落在120KB至180KB的大小范围内，则算子会保留该样本。第一个和第三个样本中至少有一张图片在这个范围内，因此它们被保留。第二个样本由于其两张图片均超出指定大小范围而被移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_size_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_size_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)