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
The operator splits videos into scene clips using the ContentDetector. It identifies 3, 1, and 2 scenes in the three input videos respectively, based on the specified threshold and minimum scene length, updating the meta data with the number of detected scenes.
算子使用ContentDetector将视频分割成场景片段。根据指定的阈值和最小场景长度，它在三个输入视频中分别识别出3、1和2个场景，并更新元数据中的检测到的场景数量。

### test_default_with_text
```python
VideoSplitBySceneMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; this is video1 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; this is video2 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | 1 video</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; this is video3 &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt;&lt;__dj__video&gt;&lt;__dj__video&gt; this is video1 &lt;|__dj__eoc|&gt;</pre><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>3</td></tr></table></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt; this is video2 &lt;|__dj__eoc|&gt;</pre><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>1</td></tr></table></details></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__video&gt;&lt;__dj__video&gt; this is video3 &lt;|__dj__eoc|&gt;</pre><details style='margin-top:6px;'><summary style='cursor:pointer;'>other key</summary><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555;'>scene_num</td><td style='padding:4px 8px;'>2</td></tr></table></details></div>

#### ✨ explanation 解释
The operator splits videos into scene clips and updates the text field to reflect the new video clips. For the first video with 3 scenes, it adds two more <__dj__video> tokens to the text. The second video remains unchanged as it has only one scene. The third video, with 2 scenes, gets an additional <__dj__video> token in its text.
算子将视频分割成场景片段，并更新文本字段以反映新的视频片段。对于有3个场景的第一个视频，在文本中添加了两个<__dj__video>标记。第二个视频只有一个场景，因此保持不变。第三个视频有2个场景，在其文本中添加了一个额外的<__dj__video>标记。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_split_by_scene_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_split_by_scene_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)