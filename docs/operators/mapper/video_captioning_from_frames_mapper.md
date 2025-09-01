# video_captioning_from_frames_mapper

Generates video captions from sampled frames using an image-to-text model. Captions from different frames are concatenated into a single string.

- Uses a Hugging Face image-to-text model to generate captions for sampled video frames.
- Supports different frame sampling methods: 'all_keyframes' or 'uniform'.
- Can apply horizontal and vertical flips to the frames before captioning.
- Offers multiple strategies for retaining generated captions: 'random_any', 'similar_one_simhash', or 'all'.
- Optionally keeps the original sample in the final dataset.
- Allows setting a global prompt or per-sample prompts to guide caption generation.
- Generates a specified number of candidate captions per video, which can be reduced based on the selected retention strategy.
- The number of output samples depends on the retention strategy and whether original samples are kept.

使用图像到文本模型从采样帧生成视频字幕。来自不同帧的字幕连接成一个字符串。

- 使用拥抱面部图像到文本模型为采样的视频帧生成字幕。
- 支持不同的帧采样方法: 'all_keyframes' 或 'uniform'。
- 可以在字幕之前将水平和垂直翻转应用于帧。
- 提供多种策略来保留生成的字幕: 'random_any' 、 'similar_one_simhash' 或 'all'。
- 可选地将原始样本保留在最终数据集中。
- 允许设置全局提示或每个样本提示以指导字幕生成。
- 为每个视频生成指定数量的候选字幕，可以根据所选的保留策略减少该数量。
- 输出样本的数量取决于保留策略以及是否保留原始样本。

Type 算子类型: **mapper**

Tags 标签: gpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_img2seq` | <class 'str'> | `'Salesforce/blip2-opt-2.7b'` | model name on huggingface to generate caption |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `caption_num` | typing.Annotated[int, Gt(gt=0)] | `1` | how many candidate captions to generate |
| `keep_candidate_mode` | <class 'str'> | `'random_any'` | retain strategy for the generated |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If |
| `prompt` | typing.Optional[str] | `None` | a string prompt to guide the generation of image-to-text |
| `prompt_key` | typing.Optional[str] | `None` | the key name of fields in samples to store prompts |
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from |
| `horizontal_flip` | <class 'bool'> | `False` | flip frame video horizontally (left to right). |
| `vertical_flip` | <class 'bool'> | `False` | flip frame video vertically (top to bottom). |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_captioning_from_frames_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_captioning_from_frames_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)