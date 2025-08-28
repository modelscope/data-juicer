# video_remove_watermark_mapper

Remove watermarks from videos based on specified regions.

This operator removes watermarks from video frames by detecting and masking the
watermark areas. It supports two detection methods: 'pixel_value' and 'pixel_diversity'.
The regions of interest (ROIs) for watermark detection can be specified as either pixel
coordinates or ratios of the frame dimensions. The operator extracts a set number of
frames uniformly from the video to detect watermark pixels. A pixel is considered part
of a watermark if it meets the detection criteria in a minimum number of frames. The
cleaned video is saved in the specified directory or the same directory as the input
file if no save directory is provided.

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `roi_strings` | typing.List[str] | `['0,0,0.1,0.1']` | a given list of regions the watermarks locate. |
| `roi_type` | <class 'str'> | `'ratio'` | the roi string type. When the type is 'pixel', (x1, |
| `roi_key` | typing.Optional[str] | `None` | the key name of fields in samples to store roi_strings |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `10` | the number of frames to be extracted uniformly from |
| `min_frame_threshold` | typing.Annotated[int, Gt(gt=0)] | `7` | a coordination is considered as the |
| `detection_method` | <class 'str'> | `'pixel_value'` | the method to detect the pixels of watermark. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_remove_watermark_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_remove_watermark_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)