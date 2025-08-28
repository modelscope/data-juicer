# video_face_blur_mapper

Mapper to blur faces detected in videos.

This operator uses an OpenCV classifier for face detection and applies a specified blur
type to the detected faces. The default classifier is 'haarcascade_frontalface_alt.xml'.
Supported blur types include 'mean', 'box', and 'gaussian'. The radius of the blur
kernel can be adjusted. If a save directory is not provided, the processed videos will
be saved in the same directory as the input files. The `DJ_PRODUCED_DATA_DIR`
environment variable can also be used to specify the save directory.

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `cv_classifier` | <class 'str'> | `''` | OpenCV classifier path for face detection. |
| `blur_type` | <class 'str'> | `'gaussian'` | Type of blur kernel, including |
| `radius` | <class 'float'> | `2` | Radius of blur kernel. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_face_blur_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_face_blur_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)