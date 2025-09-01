# optimize_response_mapper

Optimize response in question-answer pairs to be more detailed and specific.

This operator enhances the responses in question-answer pairs, making them more detailed and specific while ensuring they still address the original question. It uses a predefined system prompt for optimization. The optimized response is stripped of any leading or trailing whitespace before being returned. This mapper leverages a Hugging Face model for the optimization process, which is accelerated using CUDA.

优化问答对中的响应，使其更加详细和具体。

此运算符增强了问答对中的响应，使它们更加详细和具体，同时确保它们仍然解决原始问题。它使用预定义的系统提示进行优化。优化的响应在返回之前被去除任何前导或尾随空格。此映射器利用拥抱面模型进行优化过程，使用CUDA加速。

Type 算子类型: **mapper**

Tags 标签: cpu, vllm, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'Qwen/Qwen2.5-7B-Instruct'` | Hugging Face model ID. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the optimization task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the input for the model. |
| `qa_pair_template` | typing.Optional[str] | `None` | Template for formatting the question and |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression pattern to extract question |
| `enable_vllm` | <class 'bool'> | `False` | Whether to use VLLM for inference acceleration. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation (e.g., |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/optimize_response_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_optimize_response_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)