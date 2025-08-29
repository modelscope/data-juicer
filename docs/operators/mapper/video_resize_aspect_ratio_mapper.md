# video_resize_aspect_ratio_mapper

Resizes videos to fit within a specified aspect ratio range. This operator adjusts the
dimensions of videos to ensure their aspect ratios fall within a defined range. It can
either increase or decrease the video dimensions based on the specified strategy. The
aspect ratio is calculated as width divided by height. If a video's aspect ratio is
outside the given range, it will be resized to match the closest boundary (either the
minimum or maximum ratio). The `min_ratio` and `max_ratio` should be provided as strings
in the format "9:21" or "9/21". The resizing process uses the `ffmpeg` library to handle
the actual video scaling. Videos that do not need resizing are left unchanged. The
operator supports saving the modified videos to a specified directory or the same
directory as the input files.

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_ratio` | <class 'str'> | `'9/21'` | The minimum aspect ratio to enforce videos with |
| `max_ratio` | <class 'str'> | `'21/9'` | The maximum aspect ratio to enforce videos with |
| `strategy` | <class 'str'> | `'increase'` | The resizing strategy to apply when adjusting the |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_min_ratio_increase
```python
VideoResizeAspectRatioMapper(min_ratio='3/4', strategy='increase')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(640, 360)], [(480, 640)], [(480, 640)]]</pre></div>

#### ✨ explanation 解释
This example demonstrates how the operator resizes videos to meet a minimum aspect ratio. If a video's aspect ratio is below the specified minimum (3/4 in this case), the operator will increase the video dimensions to match the minimum ratio. In this test, the third video (originally 181:320) is resized to 3:4 (480x640). Videos that already meet or exceed the minimum ratio are not changed.
这个示例展示了算子如何调整视频以满足最小的宽高比。如果视频的宽高比低于指定的最小值（本例中为3/4），算子会增加视频的尺寸以匹配最小的宽高比。在这个测试中，第三个视频（原始宽高比为181:320）被调整为3:4（480x640）。已经满足或超过最小宽高比的视频不会被改变。

### test_max_ratio_decrease
```python
VideoResizeAspectRatioMapper(max_ratio='4/3', strategy='decrease')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(480, 360)], [(480, 640)], [(362, 640)]]</pre></div>

#### ✨ explanation 解释
This example shows the operator resizing videos to fit within a maximum aspect ratio. If a video's aspect ratio is above the specified maximum (4/3 in this case), the operator will decrease the video dimensions to match the maximum ratio. Here, the first video (originally 16:9) is resized to 4:3 (480x360). Videos that already have an aspect ratio less than or equal to the maximum are not changed.
这个示例展示了算子如何调整视频以适应最大的宽高比。如果视频的宽高比高于指定的最大值（本例中为4/3），算子会减小视频的尺寸以匹配最大宽高比。在这里，第一个视频（原始宽高比为16:9）被调整为4:3（480x360）。宽高比已经小于或等于最大值的视频不会被改变。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_resize_aspect_ratio_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_resize_aspect_ratio_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)