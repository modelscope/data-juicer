# video_watermark_filter

Filter to keep samples whose videos have no watermark with high probability.

This operator uses a Hugging Face watermark detection model to predict the probability of watermarks in video frames. It keeps samples where the predicted watermark probability is below a specified threshold. The key metric, 'video_watermark_prob', is computed by extracting frames from the video using a specified sampling method and then averaging, maximizing, or minimizing the probabilities based on the reduce mode. If multiple videos are present, the operator can use either an 'any' or 'all' strategy to determine if the sample should be kept. The frame sampling method can be 'all_keyframes' or 'uniform', and the reduce mode can be 'avg', 'max', or 'min'.



Type 算子类型: **filter**

Tags 标签: cpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_watermark_model` | <class 'str'> | `'amrul-hzz/watermark_detector'` | watermark detection model name on |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `prob_threshold` | <class 'float'> | `0.8` | the predicted watermark probability threshold |
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from |
| `reduce_mode` | <class 'str'> | `'avg'` | reduce mode for multiple sampled video frames. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_all_keyframes
```python
VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model, prob_threshold=0.8, frame_sampling_method='all_keyframes')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### ✨ explanation 解释
This operator filters out samples where the probability of watermarks in video frames, as determined by a Hugging Face model, is above 0.8. It uses all keyframes for sampling. In this case, only the sample with video3 is kept because its watermark probability is below the threshold.
该算子过滤掉视频帧中水印概率高于0.8的样本，使用Hugging Face模型进行预测，并对所有关键帧进行采样。在这种情况下，只有video3的样本被保留，因为它的水印概率低于阈值。

### test_reduce_max
```python
VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model, prob_threshold=0.9, frame_sampling_method='all_keyframes', reduce_mode='max')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[]</pre></div>

#### ✨ explanation 解释
This operator filters out samples if the maximum watermark probability among all keyframes exceeds 0.9. Here, all samples are removed since at least one keyframe in each video has a watermark probability higher than 0.9, demonstrating how 'max' reduce mode works under stricter filtering conditions.
该算子如果所有关键帧中的最大水印概率超过0.9，则会过滤掉这些样本。这里，所有的样本都被移除，因为每个视频中至少有一个关键帧的水印概率高于0.9，这展示了在更严格的过滤条件下'max'减少模式的工作方式。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_watermark_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_watermark_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)