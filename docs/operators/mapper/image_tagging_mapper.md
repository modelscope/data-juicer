# image_tagging_mapper

Generates image tags for each image in the sample.

This operator processes images to generate descriptive tags. It uses a Hugging Face
model to analyze the images and produce relevant tags. The tags are stored in the
specified field, defaulting to 'image_tags'. If the tags are already present in the
sample, the operator will not recompute them. For samples without images, an empty tag
array is assigned. The generated tags are sorted by frequency and stored as a list of
strings.

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tag_field_name` | <class 'str'> | `'image_tags'` | the field name to store the tags. It's |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_tagging_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_tagging_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)