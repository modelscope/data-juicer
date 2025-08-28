# sentence_augmentation_mapper

Augments sentences by generating enhanced versions using a Hugging Face model. This
operator enhances input sentences by generating new, augmented versions. It is designed
to work best with individual sentences rather than full documents. For optimal results,
ensure the input text is at the sentence level. The augmentation process uses a Hugging
Face model, such as `lmsys/vicuna-13b-v1.5` or `Qwen/Qwen2-7B-Instruct`. The operator
requires specifying both the primary and secondary text keys, where the augmented
sentence will be stored in the secondary key. The generation process can be customized
with parameters like temperature, top-p sampling, and beam search size.

Type 算子类型: **mapper**

Tags 标签: cpu, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'Qwen/Qwen2-7B-Instruct'` | Huggingface model id. |
| `system_prompt` | <class 'str'> | `None` | System prompt. |
| `task_sentence` | <class 'str'> | `None` | The instruction for the current task. |
| `max_new_tokens` |  | `256` | the maximum number of new tokens |
| `temperature` |  | `0.2` | used to control the randomness of |
| `top_p` |  | `None` | randomly select the next word from the group |
| `num_beams` |  | `1` | the larger the beam search size, the higher |
| `text_key` |  | `None` | the key name used to store the first sentence |
| `text_key_second` |  | `None` | the key name used to store the second sentence |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/sentence_augmentation_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_sentence_augmentation_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)