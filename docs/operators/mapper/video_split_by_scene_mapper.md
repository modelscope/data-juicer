# video_split_by_scene_mapper

Splits videos into scene clips based on detected scene changes.

This operator uses a specified scene detector to identify and split video scenes. It
supports three types of detectors: ContentDetector, ThresholdDetector, and
AdaptiveDetector. The operator processes each video in the sample, detects scenes, and
splits the video into individual clips. The minimum length of a scene can be set, and
progress can be shown during processing. The resulting clips are saved in the specified
directory or the same directory as the input files if no save directory is provided. The
operator also updates the text field in the sample to reflect the new video clips. If a
video does not contain any scenes, it remains unchanged.

Type 算子类型: **mapper**

Tags 标签: cpu, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `detector` | <class 'str'> | `'ContentDetector'` | Algorithm from `scenedetect.detectors`. Should be one |
| `threshold` | typing.Annotated[float, Ge(ge=0)] | `27.0` | Threshold passed to the detector. |
| `min_scene_len` | typing.Annotated[int, Ge(ge=0)] | `15` | Minimum length of any scene. |
| `show_progress` | <class 'bool'> | `False` | Whether to show progress from scenedetect. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_ContentDetector
```python
VideoSplitBySceneMapper(detector='ContentDetector', threshold=27.0, min_scene_len=15)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>3</td></tr></table></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> empty</div><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>1</td></tr></table></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> empty</div><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>2</td></tr></table></details></div>

#### ✨ explanation 解释
This example uses the ContentDetector to split videos into scenes. The operator processes each video and detects scene changes based on a threshold of 27.0, with a minimum scene length of 15 seconds. The result shows that the first video is split into 3 scenes, the second video has only 1 scene, and the third video is split into 2 scenes. This demonstrates how the operator can handle different types of videos and split them into multiple scenes based on content changes.
这个例子使用ContentDetector将视频分割成多个场景。算子处理每个视频，并根据阈值27.0检测场景变化，最小场景长度为15秒。结果显示，第一个视频被分割成3个场景，第二个视频只有1个场景，第三个视频被分割成2个场景。这展示了算子如何处理不同类型的视频，并根据内容变化将其分割成多个场景。

### test_default_with_text
```python
VideoSplitBySceneMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; this is video1 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; this is video2 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; this is video3 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt;&lt;__dj__video&gt;&lt;__dj__video&gt; this is video1 &lt;|__dj__eoc|&gt;</pre><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>3</td></tr></table></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; this is video2 &lt;|__dj__eoc|&gt;</pre><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>1</td></tr></table></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt;&lt;__dj__video&gt; this is video3 &lt;|__dj__eoc|&gt;</pre><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>2</td></tr></table></details></div>

#### ✨ explanation 解释
This example shows the default behavior of the VideoSplitBySceneMapper when it also updates the text field in the sample. The operator splits the videos into scenes and updates the 'text' field to reflect the new video clips. For instance, the first video is split into 3 scenes, so the 'text' field is updated to include three video tokens. This demonstrates how the operator not only splits the video but also updates the associated text to match the new video structure.
这个例子展示了VideoSplitBySceneMapper在更新样本中的文本字段时的默认行为。算子将视频分割成多个场景，并更新'text'字段以反映新的视频片段。例如，第一个视频被分割成3个场景，因此'text'字段被更新为包含三个视频标记。这展示了算子不仅分割视频，还更新相关文本以匹配新的视频结构。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_split_by_scene_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_split_by_scene_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)