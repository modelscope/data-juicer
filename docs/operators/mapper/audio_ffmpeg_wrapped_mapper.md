# audio_ffmpeg_wrapped_mapper

Wraps FFmpeg audio filters for processing audio files in a dataset.

This operator applies specified FFmpeg audio filters to the audio files in the dataset.
It supports passing custom filter parameters and global arguments to the FFmpeg command
line. The processed audio files are saved to a specified directory or the same directory
as the input files if no save directory is provided. The `DJ_PRODUCED_DATA_DIR`
environment variable can also be used to set the save directory. If no filter name is
provided, the audio files remain unmodified. The operator updates the source file paths
in the dataset after processing.

Type 算子类型: **mapper**

Tags 标签: cpu, audio

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `filter_name` | typing.Optional[str] | `None` | ffmpeg audio filter name. |
| `filter_kwargs` | typing.Optional[typing.Dict] | `None` | keyword-arguments passed to ffmpeg filter. |
| `global_args` | typing.Optional[typing.List[str]] | `None` | list-arguments passed to ffmpeg command-line. |
| `capture_stderr` | <class 'bool'> | `True` | whether to capture stderr. |
| `overwrite_output` | <class 'bool'> | `True` | whether to overwrite output file. |
| `save_dir` | <class 'str'> | `None` | The directory where generated audio files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_resize
```python
AudioFFmpegWrappedMapper('atrim', filter_kwargs={'end': 6}, capture_stderr=False)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 3 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio2.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[5.501678004535147, 6.0, 6.0]]</pre></div>

#### ✨ explanation 解释
The operator applies the 'atrim' filter with an end time of 6 seconds to each audio file, trimming any part of the audio that exceeds this duration. The resulting durations are [5.501678004535147, 6.0, 6.0] seconds, indicating that all audios have been trimmed to at most 6 seconds, with the first one being naturally shorter.
算子对每个音频文件应用了'atrim'滤镜，并设置了结束时间为6秒，从而修剪掉超过这个时长的部分。结果的时长为[5.501678004535147, 6.0, 6.0]秒，表明所有音频都被裁剪至最多6秒，其中第一个音频自然较短。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/audio_ffmpeg_wrapped_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_audio_ffmpeg_wrapped_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)