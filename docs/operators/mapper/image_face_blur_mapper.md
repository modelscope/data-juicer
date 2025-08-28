# image_face_blur_mapper

Mapper to blur faces detected in images.

This operator uses an OpenCV classifier to detect faces in images and applies a
specified blur type to the detected face regions. The blur types supported are 'mean',
'box', and 'gaussian'. The radius of the blur kernel can be adjusted. If no save
directory is provided, the modified images will be saved in the same directory as the
input files. The operator uses a Hugging Face tokenizer for text processing if needed.

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `cv_classifier` | <class 'str'> | `''` | OpenCV classifier path for face detection. |
| `blur_type` | <class 'str'> | `'gaussian'` | Type of blur kernel, including |
| `radius` | typing.Annotated[float, Ge(ge=0)] | `2` | Radius of blur kernel. |
| `save_dir` | <class 'str'> | `None` | The directory where generated image files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_face_blur_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_face_blur_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)