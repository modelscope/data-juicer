# video_ffmpeg_wrapped_mapper

Wraps FFmpeg video filters for processing video files in a dataset.

This operator applies a specified FFmpeg video filter to each video file in the dataset.
It supports passing keyword arguments to the filter and global arguments to the FFmpeg
command line. The processed videos are saved in a specified directory or the same
directory as the input files. If no filter name is provided, the videos remain
unmodified. The operator updates the source file paths in the dataset to reflect any
changes.

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `filter_name` | typing.Optional[str] | `None` | ffmpeg video filter name. |
| `filter_kwargs` | typing.Optional[typing.Dict] | `None` | keyword-arguments passed to ffmpeg filter. |
| `global_args` | typing.Optional[typing.List[str]] | `None` | list-arguments passed to ffmpeg command-line. |
| `capture_stderr` | <class 'bool'> | `True` | whether to capture stderr. |
| `overwrite_output` | <class 'bool'> | `True` | whether to overwrite output file. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_resize
```python
VideoFFmpegWrappedMapper('scale', filter_kwargs={'width': 400, 'height': 480}, capture_stderr=False)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 3 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4|video2.mp4|video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(400, 480), (400, 480), (400, 480)]]</pre></div>

#### ✨ explanation 解释
This example demonstrates the use of the VideoFFmpegWrappedMapper operator to resize videos. The 'scale' filter is applied with a specified width and height, resulting in all videos being resized to 400x480 pixels. The original aspect ratio of the videos may be altered to fit the new dimensions. In this case, the output data shows that all three input videos have been resized to 400x480, as expected.
此示例展示了使用VideoFFmpegWrappedMapper算子调整视频大小的功能。通过应用'scale'滤镜并指定宽度和高度，所有视频都被调整为400x480像素。原始视频的宽高比可能会被改变以适应新的尺寸。这里的结果数据显示，三个输入视频都已成功调整到了400x480，符合预期。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_ffmpeg_wrapped_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_ffmpeg_wrapped_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)