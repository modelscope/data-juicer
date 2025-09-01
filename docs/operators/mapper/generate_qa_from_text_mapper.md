# generate_qa_from_text_mapper

Generates question and answer pairs from text using a specified model.

This operator uses a Hugging Face model to generate QA pairs from the input text. It supports both Hugging Face and vLLM models for inference. The recommended models, such as 'alibaba-pai/pai-llama3-8b-doc2qa', are trained on Chinese data and are suitable for Chinese text. The operator can limit the number of generated QA pairs per text and allows custom output patterns for parsing the model's response. By default, it uses a regular expression to extract questions and answers from the model's output. If no QA pairs are extracted, a warning is logged.

使用指定的模型从文本生成问题和答案对。

此运算符使用拥抱面模型从输入文本生成QA对。它支持拥抱面和vLLM模型进行推理。推荐的模型，如 “阿里巴巴-派/pai-llama3-8b-doc2qa”，是在中文数据上训练的，适用于中文文本。运算符可以限制每个文本生成的QA对的数量，并允许自定义输出模式来解析模型的响应。默认情况下，它使用正则表达式从模型的输出中提取问题和答案。如果未提取QA对，则记录警告。

Type 算子类型: **mapper**

Tags 标签: cpu, vllm, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'alibaba-pai/pai-qwen1_5-7b-doc2qa'` | Huggingface model ID. |
| `max_num` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The max num of returned QA sample for each text. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression pattern to extract |
| `enable_vllm` | <class 'bool'> | `False` | Whether to use vllm for inference acceleration. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation, |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/generate_qa_from_text_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_generate_qa_from_text_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)