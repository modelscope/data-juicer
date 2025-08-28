# video_tagging_from_audio_mapper

Generates video tags from audio streams using the Audio Spectrogram Transformer.

This operator extracts audio streams from videos and uses a Hugging Face Audio
Spectrogram Transformer (AST) model to generate tags. The tags are stored in the
specified metadata field, defaulting to 'video_audio_tags'. If no valid audio stream is
found, the tag is set to 'EMPTY'. The operator resamples audio to match the model's
required sampling rate if necessary. The tags are inferred based on the highest logit
value from the model's output. If the tags are already present in the sample, the
operator skips processing for that sample.

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
VideoTaggingFromAudioMapper(self.hf_ast)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。&lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 一个人在帮另一个人梳头发。 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video4.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video4.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; 一个穿着红色连衣裙的女人在试衣服。 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video5.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video5.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[&#x27;Music&#x27;], [&#x27;Music&#x27;], [&#x27;Speech&#x27;], [&#x27;Speech&#x27;]]</pre></div>

#### ✨ explanation 解释
The operator extracts audio from each video and uses a Hugging Face Audio Spectrogram Transformer (AST) model to generate tags. The tags 'Music' or 'Speech' are assigned based on the highest logit value from the model's output. In this case, the first two videos are tagged as 'Music', and the last two as 'Speech'.
算子从每个视频中提取音频，并使用Hugging Face的音频频谱图转换器（AST）模型生成标签。根据模型输出的最大logit值，为视频分配'Music'或'Speech'标签。在这种情况下，前两个视频被标记为'Music'，后两个被标记为'Speech'。

### test_no_audio
```python
VideoTaggingFromAudioMapper(self.hf_ast)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 3 videos</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; &lt;__dj__video&gt; 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪; 两个长头发的女子正坐在一张圆桌前讲话互动。 &lt;|__dj__eoc|&gt;&lt;__dj__video&gt; 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4|video3-no-audio.mp4|video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video3-no-audio.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 3 videos</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; &lt;__dj__video&gt; 两个长头发的女子正坐在一张圆桌前讲话互动。 &lt;__dj__video&gt; 一个人在帮另一个人梳头发。</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4|video3-no-audio.mp4|video4.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video3-no-audio.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video4.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[&#x27;Music&#x27;, &#x27;EMPTY&#x27;, &#x27;Music&#x27;], [&#x27;Music&#x27;, &#x27;EMPTY&#x27;, &#x27;Speech&#x27;]]</pre></div>

#### ✨ explanation 解释
When there is no valid audio stream in a video, the operator assigns the tag 'EMPTY' to that video. In this example, one of the videos in both samples lacks an audio stream, resulting in the 'EMPTY' tag being assigned. The other videos are tagged as 'Music' or 'Speech' based on their content.
当视频中没有有效的音频流时，算子将为该视频分配'EMPTY'标签。在这个例子中，两个样本中的一个视频都缺少音频流，因此被赋予了'EMPTY'标签。其他视频则根据其内容被标记为'Music'或'Speech'。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_tagging_from_audio_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_tagging_from_audio_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)