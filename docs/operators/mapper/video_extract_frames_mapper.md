# video_extract_frames_mapper

Mapper to extract frames from video files according to specified methods.

- Extracts frames based on the chosen method: 'all_keyframes' or 'uniform'.
- For 'all_keyframes', extracts all keyframes. For 'uniform', extracts a specified
number of frames uniformly.
- If 'duration' is set, the video is segmented, and frames are extracted from each
segment.
- The extracted frames are saved in a directory, and the paths are stored in a
dictionary.
- The dictionary maps video keys to their respective frame directories.
- If 'frame_dir' is not provided, a default directory based on the video file path is
used.
- The resulting dictionary is saved under the specified 'frame_key' in the sample's
metadata.

Type 算子类型: **mapper**

Tags 标签: cpu, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from |
| `duration` | <class 'float'> | `0` | The duration of each segment in seconds. |
| `frame_dir` | <class 'str'> | `None` | Output directory to save extracted frames. |
| `frame_key` |  | `'video_frames'` | The name of field to save generated frames info. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_extract_frames_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_extract_frames_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)