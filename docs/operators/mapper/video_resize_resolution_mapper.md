# video_resize_resolution_mapper

Resizes video resolution based on specified width and height constraints.

This operator resizes videos to fit within the provided minimum and maximum width and
height limits. It can optionally maintain the original aspect ratio by adjusting the
dimensions accordingly. The resized videos are saved in the specified directory or the
same directory as the input if no save directory is provided. The key metric for
resizing is the video's width and height, which are adjusted to meet the constraints
while maintaining the aspect ratio if configured. The `force_divisible_by` parameter
ensures that the output dimensions are divisible by a specified integer, which must be a
positive even number when used with aspect ratio adjustments.

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_width` | <class 'int'> | `1` | Videos with width less than 'min_width' will be |
| `max_width` | <class 'int'> | `9223372036854775807` | Videos with width more than 'max_width' will be |
| `min_height` | <class 'int'> | `1` | Videos with height less than 'min_height' will be |
| `max_height` | <class 'int'> | `9223372036854775807` | Videos with height more than 'max_height' will be |
| `force_original_aspect_ratio` | <class 'str'> | `'disable'` | Enable decreasing or             increasing output video width or height if necessary             to keep the original aspect ratio, including ['disable',             'decrease', 'increase']. |
| `force_divisible_by` | typing.Annotated[int, Gt(gt=0)] | `2` | Ensures that both the output dimensions,             width and height, are divisible by the given integer when used             together with force_original_aspect_ratio, must be a positive             even number. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_default_mapper
```python
VideoResizeResolutionMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(640, 360)], [(480, 640)], [(362, 640)]]</pre></div>

#### ✨ explanation 解释
The operator resizes the videos to fit within default width and height limits, maintaining their original aspect ratios. The resulting dimensions for each video are (640, 360), (480, 640), and (362, 640) respectively, as they are adjusted to meet the constraints while keeping the aspect ratio unchanged.
算子将视频调整为默认的宽度和高度限制内，同时保持原始纵横比。每个视频调整后的尺寸分别为(640, 360)，(480, 640) 和 (362, 640)，这些调整确保了在满足约束的同时保持纵横比不变。

### test_force_divisible_by
```python
VideoResizeResolutionMapper(min_width=400, max_width=480, min_height=480, max_height=480, force_original_aspect_ratio='decrease', force_divisible_by=4)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(480, 272)]]</pre></div>

#### ✨ explanation 解释
The operator resizes the video with a specific constraint that the output dimensions must be divisible by 4, while also decreasing the size to fit within the given width and height limits. The result is a video resized to (480, 272), which meets the divisibility requirement and fits within the specified constraints.
算子调整视频尺寸时要求输出的尺寸必须能够被4整除，同时减小尺寸以适应给定的宽度和高度限制。结果是视频被调整到了(480, 272)，这既满足了可整除性要求也符合指定的约束条件。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_resize_resolution_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_resize_resolution_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)