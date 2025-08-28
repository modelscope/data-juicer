# image_captioning_mapper

Generates image captions using a Hugging Face model and appends them to samples.

This operator generates captions for images in the input samples using a specified
Hugging Face model. It can generate multiple captions per image and apply different
strategies to retain the generated captions. The operator supports three retention
modes: 'random_any', 'similar_one_simhash', and 'all'. In 'random_any' mode, a random
caption is retained. In 'similar_one_simhash' mode, the most similar caption to the
original text (based on SimHash) is retained. In 'all' mode, all generated captions are
concatenated and retained. The operator can also keep or discard the original sample
based on the `keep_original_sample` parameter. If both `prompt` and `prompt_key` are
set, the `prompt_key` takes precedence.

Type 算子类型: **mapper**

Tags 标签: cpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_img2seq` | <class 'str'> | `'Salesforce/blip2-opt-2.7b'` | model name on huggingface to generate caption |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `caption_num` | typing.Annotated[int, Gt(gt=0)] | `1` | how many candidate captions to generate |
| `keep_candidate_mode` | <class 'str'> | `'random_any'` | retain strategy for the generated |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If |
| `prompt` | typing.Optional[str] | `None` | a string prompt to guide the generation of blip2 model |
| `prompt_key` | typing.Optional[str] | `None` | the key name of fields in samples to store prompts |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_captioning_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_captioning_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)