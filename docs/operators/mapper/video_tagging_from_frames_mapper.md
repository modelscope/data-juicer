# video_tagging_from_frames_mapper

Generates video tags from frames extracted from videos.

This operator extracts frames from videos and generates tags based on the
content of these frames. The frame extraction method can be either
"all_keyframes" or "uniform". For "all_keyframes", all keyframes are
extracted, while for "uniform", a specified number of frames are
extracted uniformly across the video. The tags are generated using a
pre-trained model and stored in the specified field name. If the tags
are already present in the sample, the operator skips processing.
Important notes:
- Uses a Hugging Face tokenizer and a pre-trained model for tag generation.
- If no video is present in the sample, an empty tag array is stored.
- Frame tensors are processed to generate tags, which are then sorted by
frequency and stored.

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from |
| `tag_field_name` | <class 'str'> | `'video_frame_tags'` | the field name to store the tags. It's |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_tagging_from_frames_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_tagging_from_frames_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)