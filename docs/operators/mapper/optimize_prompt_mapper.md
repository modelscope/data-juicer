# optimize_prompt_mapper


Mapper to optimize prompts based on the existing ones. This OP will use the existing prompts in the same batch and newly optimized prompts as the examples to optimize the next ones.

Reference: https://doc.agentscope.io/v0/en/build_tutorial/prompt_optimization.html


用于根据现有提示进行优化的映射器。此算子将使用同一批次中的现有提示和新优化的提示作为示例来优化后续的提示。

参考：https://doc.agentscope.io/v0/en/build_tutorial/prompt_optimization.html


*****

Type 算子类型: **mapper**

Tags 标签: cpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'Qwen/Qwen2.5-7B-Instruct'` | API or huggingface model name. |
| `gen_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of new prompts to generate. |
| `max_example_num` | typing.Annotated[int, Gt(gt=0)] | `3` |  |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated texts in the final datasets and the original texts will be removed. It's True in default. |
| `retry_num` | <class 'int'> | `3` | how many times to retry to generate the prompt if the parsed generated prompt is empty. It's 3 in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the generation task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the input prompt. It must include one placeholder '{}', which will be replaced by `example_num` formatted examples defined by `example_template`. |
| `example_template` | typing.Optional[str] | `None` | Template for formatting one prompt example. It must include one placeholder '{}', which will be replaced by one formatted prompt. |
| `prompt_template` | typing.Optional[str] | `None` | Template for formatting a single prompt within each example. Must include two placeholders '{}' for the question and answer. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression pattern to extract questions and answers from model response. |
| `enable_vllm` | <class 'bool'> | `False` | Whether to use vllm for inference acceleration. |
| `is_hf_model` | <class 'bool'> | `False` | If true, use Transformers for loading hugging face or local llm. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/optimize_prompt_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_optimize_prompt_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)