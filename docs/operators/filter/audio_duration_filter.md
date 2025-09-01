# audio_duration_filter

Keep data samples whose audio durations are within a specified range.

This operator filters data samples based on the duration of their audio files. It keeps samples where the audio duration is between a minimum and maximum value, in seconds. The operator supports two strategies for keeping samples: 'any' (keep if any audio meets the condition) or 'all' (keep only if all audios meet the condition). The audio duration is computed using the `librosa` library. If the audio duration has already been computed, it is retrieved from the sample's stats under the key 'audio_duration'. If no audio is present in the sample, an empty array is stored in the stats.

保留音频持续时间在指定范围内的数据样本。

该运算符根据音频文件的持续时间过滤数据样本。它保留音频持续时间在最小值和最大值之间的样本，以秒为单位。运营商支持两种保持样本的策略: “任何” (如果任何音频满足条件则保持) 或 “全部” (仅当所有音频满足条件时保持)。使用 'librosa' 库计算音频持续时间。如果已经计算了音频持续时间，则从关键字 “audio_duration” 下的样本统计信息中检索音频持续时间。如果样本中没有音频，则会在统计信息中存储一个空数组。

Type 算子类型: **filter**

Tags 标签: cpu, audio

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_duration` | <class 'int'> | `0` | The min audio duration to keep samples in seconds. |
| `max_duration` | <class 'int'> | `9223372036854775807` | The max audio duration to keep samples in seconds. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_filter_audios_within_range
```python
AudioDurationFilter(min_duration=10, max_duration=20)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### ✨ explanation 解释
The operator filters out audio samples that do not have a duration between 10 and 20 seconds. Only the sample with 'audio2.wav' meets this condition, hence it is the only one kept in the target list.
算子过滤掉音频时长不在10到20秒之间的样本。只有'audio2.wav'满足这个条件，因此目标列表中只保留了这一个样本。

### test_any
```python
AudioDurationFilter(min_duration=10, max_duration=20, any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### ✨ explanation 解释
The operator is set to keep any sample that has at least one audio file within the 10 to 20 seconds range. Both the first and second samples contain at least one audio meeting the condition, so they are kept. The third sample does not meet the criteria and is thus removed from the target list.
算子设置为只要有一个音频文件的时长在10到20秒之间就保留该样本。第一个和第二个样本中至少有一个音频满足条件，所以它们被保留下来。第三个样本不满足条件，因此从目标列表中移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/audio_duration_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_audio_duration_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)