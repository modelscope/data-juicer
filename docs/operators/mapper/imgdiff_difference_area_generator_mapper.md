# imgdiff_difference_area_generator_mapper

Generates and filters bounding boxes for image pairs based on similarity, segmentation,
and text matching.

This operator processes image pairs to identify and filter regions with significant
differences. It uses a sequence of operations:
- Filters out image pairs with large differences.
- Segments the images to identify potential objects.
- Crops sub-images based on bounding boxes.
- Determines if the sub-images contain valid objects using image-text matching.
- Filters out sub-images that are too similar.
- Removes overlapping bounding boxes.
- Uses Hugging Face models for similarity and text matching, and FastSAM for
segmentation.
- Caches intermediate results in `DATA_JUICER_ASSETS_CACHE`.
- Returns the filtered bounding boxes in the `MetaKeys.bbox_tag` field.

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `image_pair_similarity_filter_args` | typing.Optional[typing.Dict] | `{}` |  |
| `image_segment_mapper_args` | typing.Optional[typing.Dict] | `{}` |  |
| `image_text_matching_filter_args` | typing.Optional[typing.Dict] | `{}` |  |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/imgdiff_difference_area_generator_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_imgdiff_difference_area_generator_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)