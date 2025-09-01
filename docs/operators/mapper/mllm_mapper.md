# mllm_mapper

Mapper to use MLLMs for visual question answering tasks. This operator uses a Hugging Face model to generate answers based on input text and images. It supports models like `llava-hf/llava-v1.6-vicuna-7b-hf` and `Qwen/Qwen2-VL-7B-Instruct`. The operator processes each sample, loading and processing images, and generating responses using the specified model. The generated responses are appended to the sample's text field. The key parameters include the model ID, maximum new tokens, temperature, top-p sampling, and beam search size, which control the generation process.

Mapper使用MLLMs进行视觉问答任务。该操作员使用拥抱面部模型来基于输入文本和图像生成答案。它支持 “lava-hf/llava-v1.6-vicuna-7b-hf” 和 “qwen/Qwen2-VL-7B-Instruct” 等型号。操作员处理每个样本，加载和处理图像，并使用指定的模型生成响应。生成的响应将附加到示例的文本字段。关键参数包括控制生成过程的模型ID、最大新令牌、温度、top-p采样和光束搜索大小。

Type 算子类型: **mapper**

Tags 标签: cpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'llava-hf/llava-v1.6-vicuna-7b-hf'` | hugginface model id. |
| `max_new_tokens` |  | `256` | the maximum number of new tokens |
| `temperature` |  | `0.2` | used to control the randomness of             generated text. The higher the temperature, the more                 random and creative the generated text will be. |
| `top_p` |  | `None` | randomly select the next word from the group             of words whose cumulative probability reaches p. |
| `num_beams` |  | `1` | the larger the beam search size, the higher             the quality of the generated text. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/mllm_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_mllm_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)