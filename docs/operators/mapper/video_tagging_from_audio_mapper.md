# video_tagging_from_audio_mapper

Generates video tags from audio streams using the Audio Spectrogram Transformer.

This operator extracts audio streams from videos and uses a Hugging Face Audio Spectrogram Transformer (AST) model to generate tags. The tags are stored in the specified metadata field, defaulting to 'video_audio_tags'. If no valid audio stream is found, the tag is set to 'EMPTY'. The operator resamples audio to match the model's required sampling rate if necessary. The tags are inferred based on the highest logit value from the model's output. If the tags are already present in the sample, the operator skips processing for that sample.

使用音频频谱图转换器从音频流生成视频标签。

该运算符从视频中提取音频流，并使用拥抱面部音频频谱图转换器 (AST) 模型来生成标签。标签存储在指定的元数据字段中，默认为 “video_audio_tags”。如果没有找到有效的音频流，则将标签设置为 “空”。如有必要，操作员将对音频进行重采样以匹配模型所需的采样率。根据模型输出的最高logit值推断标记。如果标签已经存在于样本中，则操作者跳过对该样本的处理。

Type 算子类型: **mapper**

Tags 标签: cpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_ast` | <class 'str'> | `'MIT/ast-finetuned-audioset-10-10-0.4593'` | path to the HF model to tag from audios. |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models |
| `tag_field_name` | <class 'str'> | `'video_audio_tags'` | the field name to store the tags. It's |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test
```python
VideoTaggingFromAudioMapper('MIT/ast-finetuned-audioset-10-10-0.4593')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。&lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 一个人在帮另一个人梳头发。 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video4.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video4.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 一个穿着红色连衣裙的女人在试衣服。 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video5.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video5.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[&#x27;Music&#x27;], [&#x27;Music&#x27;], [&#x27;Speech&#x27;], [&#x27;Speech&#x27;]]</pre></div>

#### ✨ explanation 解释
This example demonstrates the typical usage of the operator, where it processes a list of video samples and generates audio tags. The operator extracts the audio from each video, uses the Audio Spectrogram Transformer (AST) model to analyze the audio, and then assigns a tag ('Music' or 'Speech') based on the highest logit value. The output data shows the assigned tags for each sample, indicating that the first two videos are tagged as 'Music' and the last two as 'Speech'.
该示例展示了算子的典型用法，它处理一系列视频样本并生成音频标签。算子从每个视频中提取音频，使用音频频谱图变换器（AST）模型分析音频，然后根据最高的logit值分配一个标签（'音乐'或'语音'）。输出数据显示了为每个样本分配的标签，表明前两个视频被标记为'音乐'，后两个被标记为'语音'。

### test_no_audio
```python
VideoTaggingFromAudioMapper('MIT/ast-finetuned-audioset-10-10-0.4593')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 3 videos</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; &lt;__dj__video&gt; 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪; 两个长头发的女子正坐在一张圆桌前讲话互动。 &lt;|__dj__eoc|&gt;&lt;__dj__video&gt; 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +2 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 2 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3-no-audio.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 3 videos</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; &lt;__dj__video&gt; 两个长头发的女子正坐在一张圆桌前讲话互动。 &lt;__dj__video&gt; 一个人在帮另一个人梳头发。</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4 +2 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 2 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3-no-audio.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video4.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[&#x27;Music&#x27;, &#x27;EMPTY&#x27;, &#x27;Music&#x27;], [&#x27;Music&#x27;, &#x27;EMPTY&#x27;, &#x27;Speech&#x27;]]</pre></div>

#### ✨ explanation 解释
This example illustrates an important edge case where some videos do not have an audio stream. In such cases, the operator still processes the videos but assigns the 'EMPTY' tag to those without valid audio. This ensures that all videos, even those with missing audio, are accounted for in the dataset. The output data shows that the second video in the first sample and the second video in the second sample are tagged as 'EMPTY', while the rest are tagged as 'Music' or 'Speech' based on their audio content.
该示例展示了一个重要的边缘情况，即某些视频没有音频流。在这种情况下，算子仍然处理这些视频，但对那些没有有效音频的视频分配'EMPTY'标签。这确保了所有视频，即使那些缺少音频的视频，在数据集中也被考虑在内。输出数据显示第一个样本中的第二个视频和第二个样本中的第二个视频被标记为'EMPTY'，而其余的则根据其音频内容被标记为'音乐'或'语音'。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_tagging_from_audio_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_tagging_from_audio_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)