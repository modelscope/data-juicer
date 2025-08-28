# image_captioning_from_gpt4v_mapper

Generates text captions for images using the GPT-4 Vision model.

This operator generates text based on the provided images and specified parameters. It
supports different modes of text generation, including 'reasoning', 'description',
'conversation', and 'custom'. The generated text can be added to the original sample or
replace it, depending on the `keep_original_sample` parameter. The operator uses a
Hugging Face tokenizer and the GPT-4 Vision API to generate the text. The `any_or_all`
parameter determines whether all or any of the images in a sample must meet the
generation criteria for the sample to be kept. If `user_prompt_key` is set, it will use
the prompt from the sample; otherwise, it will use the `user_prompt` parameter. If both
are set, `user_prompt_key` takes precedence.

Type 算子类型: **mapper**

Tags 标签: cpu, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `mode` | <class 'str'> | `'description'` | mode of text generated from images, can be one of |
| `api_key` | <class 'str'> | `''` | the API key to authenticate the request. |
| `max_token` | <class 'int'> | `500` | the maximum number of tokens to generate. |
| `temperature` | typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=1)])] | `1.0` | controls the randomness of the output (range |
| `system_prompt` | <class 'str'> | `''` | a string prompt used to set the context of a |
| `user_prompt` | <class 'str'> | `''` | a string prompt to guide the generation of |
| `user_prompt_key` | typing.Optional[str] | `None` | the key name of fields in samples to store |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_captioning_from_gpt4v_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_captioning_from_gpt4v_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)