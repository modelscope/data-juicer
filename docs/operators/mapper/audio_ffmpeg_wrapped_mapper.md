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
The operator trims the audio files to a maximum duration of 6 seconds. If an audio file is shorter than 6 seconds, it remains unchanged. In this case, the first audio file is already less than 6 seconds long, so its duration does not change. The second and third audio files are longer than 6 seconds, so they are trimmed to exactly 6 seconds. The output data shows the durations of the processed audio files, which now all have a maximum duration of 6 seconds.
算子将音频文件裁剪到最多6秒的长度。如果音频文件短于6秒，则保持不变。在这种情况下，第一个音频文件已经不到6秒长，因此其持续时间没有变化。第二和第三个音频文件超过6秒，所以它们被裁剪成正好6秒。输出数据显示了处理后的音频文件的时长，现在所有音频文件的最大时长都为6秒。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/audio_ffmpeg_wrapped_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_audio_ffmpeg_wrapped_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)