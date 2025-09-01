# imgdiff_difference_caption_generator_mapper

Generates difference captions for bounding box regions in two images.

This operator processes pairs of images and generates captions for the differences in their bounding box regions. It uses a multi-step process:
- Describes the content of each bounding box region using a Hugging Face model.
- Crops the bounding box regions from both images.
- Checks if the cropped regions match the generated captions.
- Determines if there are differences between the two captions.
- Marks the difference area with a red box.
- Generates difference captions for the marked areas.
- The key metric is the similarity score between the captions, computed using a CLIP model.
- If no valid bounding boxes or differences are found, it returns empty captions and zeroed bounding boxes.
- Uses 'cuda' as the accelerator if any of the fused operations support it.
- Caches temporary images during processing and clears them afterward.

为两个图像中的边界框区域生成差异字幕。

该运算符处理图像对，并针对其边界框区域中的差异生成字幕。它使用多步骤过程:
- 使用拥抱人脸模型描述每个边界框区域的内容。
- 从两个图像中裁剪边界框区域。
- 检查裁剪的区域是否与生成的字幕匹配。
- 确定两个字幕之间是否存在差异。
- 用红色框标记差异区域。
- 生成标记区域的差异字幕。
- 关键度量是使用剪辑模型计算的字幕之间的相似性得分。
- 如果没有有效的边界框或差异被发现，它返回空的标题和归零的边界框。
- 使用 'cuda' 作为加速器，如果任何融合的操作支持它。
- 在处理过程中缓存临时图像，然后清除它们。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `mllm_mapper_args` | typing.Optional[typing.Dict] | `{}` | Arguments for multimodal language model mapper. |
| `image_text_matching_filter_args` | typing.Optional[typing.Dict] | `{}` | Arguments for image-text matching filter. |
| `text_pair_similarity_filter_args` | typing.Optional[typing.Dict] | `{}` | Arguments for text pair similarity filter. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/imgdiff_difference_caption_generator_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_imgdiff_difference_caption_generator_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)