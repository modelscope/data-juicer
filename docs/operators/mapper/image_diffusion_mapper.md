# image_diffusion_mapper

Generate images using a diffusion model based on provided captions.

This operator uses a Hugging Face diffusion model to generate images from given
captions. It supports different modes for retaining generated samples, including random
selection, similarity-based selection, and retaining all. The operator can also generate
captions if none are provided, using a Hugging Face image-to-sequence model. The
strength parameter controls the extent of transformation from the reference image, and
the guidance scale influences how closely the generated images match the text prompt.
Generated images can be saved in a specified directory or the same directory as the
input files. This is a batched operation, processing multiple samples at once and
producing a specified number of augmented images per sample.

Type 算子类型: **mapper**

Tags 标签: cpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_diffusion` | <class 'str'> | `'CompVis/stable-diffusion-v1-4'` | diffusion model name on huggingface to generate |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `torch_dtype` | <class 'str'> | `'fp32'` | the floating point type used to load the diffusion |
| `revision` | <class 'str'> | `'main'` | The specific model version to use. It can be a |
| `strength` | typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=1)])] | `0.8` | Indicates extent to transform the reference image. |
| `guidance_scale` | <class 'float'> | `7.5` | A higher guidance scale value encourages the |
| `aug_num` | typing.Annotated[int, Gt(gt=0)] | `1` | The image number to be produced by stable-diffusion |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If |
| `caption_key` | typing.Optional[str] | `None` | the key name of fields in samples to store captions |
| `hf_img2seq` | <class 'str'> | `'Salesforce/blip2-opt-2.7b'` | model name on huggingface to generate caption if |
| `save_dir` | <class 'str'> | `None` | The directory where generated image files will be stored. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_diffusion_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_diffusion_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)