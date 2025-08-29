# sdxl_prompt2prompt_mapper

Generates pairs of similar images using the SDXL model.

This operator uses a Hugging Face diffusion model to generate image pairs based on two
text prompts. The quality and similarity of the generated images are controlled by
parameters such as `num_inference_steps` and `guidance_scale`. The first and second text
prompts are specified using `text_key` and `text_key_second`, respectively. The
generated images are saved in the specified `output_dir` with unique filenames. The
operator requires both text keys to be set for processing.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_diffusion` | <class 'str'> | `'stabilityai/stable-diffusion-xl-base-1.0'` | diffusion model name on huggingface to generate |
| `trust_remote_code` |  | `False` |  |
| `torch_dtype` | <class 'str'> | `'fp32'` | the floating point type used to load the diffusion |
| `num_inference_steps` | <class 'float'> | `50` | The larger the value, the better the |
| `guidance_scale` | <class 'float'> | `7.5` | A higher guidance scale value encourages the |
| `text_key` |  | `None` | the key name used to store the first caption |
| `text_key_second` |  | `None` | the key name used to store the second caption |
| `output_dir` |  | `'/home/cmgzn/.cache/data_juicer/assets'` | the storage location of the generated images. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/sdxl_prompt2prompt_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_sdxl_prompt2prompt_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)