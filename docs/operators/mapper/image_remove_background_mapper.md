# image_remove_background_mapper

Mapper to remove the background of images.

This operator processes each image in the sample, removing its background. It uses the
`rembg` library to perform the background removal. If `alpha_matting` is enabled, it
applies alpha matting with specified thresholds and erosion size. The resulting images
are saved in PNG format. The `bgcolor` parameter can be set to specify a custom
background color for the cutout image. The processed images are stored in the directory
specified by `save_dir`, or in the same directory as the input files if `save_dir` is
not provided. The `source_file` field in the sample is updated to reflect the new file
paths.

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `alpha_matting` | <class 'bool'> | `False` | (bool, optional) |
| `alpha_matting_foreground_threshold` | <class 'int'> | `240` | (int, optional) |
| `alpha_matting_background_threshold` | <class 'int'> | `10` | (int, optional) |
| `alpha_matting_erode_size` | <class 'int'> | `10` | (int, optional) |
| `bgcolor` | typing.Optional[typing.Tuple[int, int, int, int]] | `None` | (Optional[Tuple[int, int, int, int]], optional) |
| `save_dir` | <class 'str'> | `None` | The directory where generated image files will be stored. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_remove_background_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_remove_background_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)