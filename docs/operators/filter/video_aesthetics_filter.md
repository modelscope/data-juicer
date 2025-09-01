# video_aesthetics_filter

Filter to keep data samples with aesthetics scores for specified frames in the videos within a specific range.

This operator evaluates the aesthetic quality of video frames using a Hugging Face model. It keeps samples where the aesthetics scores of the specified frames fall within a given range. The key metric, 'video_frames_aesthetics_score', is computed by averaging, taking the max, or min of the frame scores, depending on the reduce mode. Frame sampling can be done uniformly or by extracting all keyframes. The filter applies a 'any' or 'all' strategy to decide if a sample should be kept based on the scores of multiple videos.

过滤器将视频中指定帧的美学得分数据样本保留在特定范围内。

该操作员使用拥抱面部模型评估视频帧的美学质量。它保留指定帧的美学分数在给定范围内的样本。关键度量 “video_frames_aesthtics_score” 是通过取帧得分的最大值或最小值进行平均来计算的，具体取决于reduce模式。帧采样可以统一完成，也可以通过提取所有关键帧来完成。过滤器应用 “任何” 或 “所有” 策略，以基于多个视频的分数来决定是否应保留样本。

Type 算子类型: **filter**

Tags 标签: cpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_scorer_model` | <class 'str'> | `''` | Huggingface model name for the aesthetics |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `min_score` | <class 'float'> | `0.4` | Min score for the predicted aesthetics in a video. |
| `max_score` | <class 'float'> | `1.0` | Max score for the predicted aesthetics in a video. |
| `frame_sampling_method` | <class 'str'> | `'uniform'` | sampling method of extracting frame |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from |
| `any_or_all` | <class 'str'> | `'any'` | Keep this sample with 'any' or 'all' strategy of |
| `reduce_mode` | <class 'str'> | `'avg'` | reduce mode when one sample corresponds to |
| `args` |  | `''` | Extra positional arguments. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_aesthetics_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_aesthetics_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)