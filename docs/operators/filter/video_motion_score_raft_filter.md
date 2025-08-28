# video_motion_score_raft_filter

Filter to keep samples with video motion scores within a specified range.

This operator uses the RAFT (Recurrent All-Pairs Field Transforms) model from
torchvision to predict optical flow between video frames. It keeps samples where the
video motion score is within the given min and max score range. The motion score is
computed based on the optical flow between frames, which is estimated using the RAFT
model. The operator can sample frames at a specified FPS and apply transformations to
the frames before computing the flow.

- The RAFT model is used to estimate the optical flow.
- Frames are preprocessed using a series of transformations including normalization and
color channel flipping.
- The motion score is calculated from the optical flow data.
- The operator can be configured to filter based on any or all frames in the video.
- The device for model inference (CPU or CUDA) is automatically detected and set.

Type 算子类型: **filter**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_score` | <class 'float'> | `1.0` |  |
| `max_score` | <class 'float'> | `1.7976931348623157e+308` |  |
| `sampling_fps` | typing.Annotated[float, Gt(gt=0)] | `2` |  |
| `size` | typing.Union[typing.Annotated[int, Gt(gt=0)], typing.Tuple[typing.Annotated[int, Gt(gt=0)]], typing.Tuple[typing.Annotated[int, Gt(gt=0)], typing.Annotated[int, Gt(gt=0)]], NoneType] | `None` |  |
| `max_size` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` |  |
| `divisible` | typing.Annotated[int, Gt(gt=0)] | `8` |  |
| `relative` | <class 'bool'> | `False` |  |
| `any_or_all` | <class 'str'> | `'any'` |  |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
### test_middle
```python
VideoMotionScoreRaftFilter(min_score=3, max_score=10.2)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### ✨ explanation 解释
The operator filters videos based on a motion score range, keeping only those with scores between 3 and 10.2. In this case, only the second video meets the criteria, as its motion score falls within the specified range, while the other videos' scores do not.
算子根据运动得分范围过滤视频，只保留得分在3到10.2之间的视频。在这种情况下，只有第二个视频符合标准，因为它的运动得分落在指定范围内，而其他视频的得分则不在该范围内。

### test_all
```python
VideoMotionScoreRaftFilter(min_score=3, max_score=10.2, any_or_all='all')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4|video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4|video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4|video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[]</pre></div>

#### ✨ explanation 解释
The operator is configured to keep samples where all videos in each sample have a motion score within the specified range (3 to 10.2). Since no sample in the input has all videos meeting this criterion, the result is an empty list, indicating that none of the samples are kept.
算子被配置为保留每个样本中所有视频的运动得分都在指定范围（3到10.2）内的样本。由于输入中的没有一个样本满足这一条件，结果是一个空列表，表明没有任何样本被保留。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_motion_score_raft_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_motion_score_raft_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)