# nlpcda_zh_mapper

Augments Chinese text samples using the nlpcda library.

This operator applies various augmentation methods to Chinese text, such as replacing similar words, homophones, deleting random characters, swapping characters, and replacing equivalent numbers. The number of augmented samples generated can be controlled by the `aug_num` parameter. If `sequential` is set to True, the augmentation methods are applied in sequence; otherwise, they are applied independently. The original sample can be kept or removed based on the `keep_original_sample` flag. It is recommended to use 1-3 augmentation methods at a time to avoid significant changes in the semantics of the samples. Some augmentation methods may not work for special texts, resulting in no augmented samples being generated.

使用nlpcda库扩充中文文本样本。

此运算符将各种增强方法应用于中文文本，例如替换相似的单词，同音字，删除随机字符，交换字符以及替换等效数字。生成的增广样本的数量可以由 '8月' 参数控制。如果 “sequential' 设置为True，则按顺序应用增强方法; 否则，独立应用它们。可以基于 'keep_original_sample' 标志来保持或移除原始样本。建议一次使用1-3种增强方法，以避免样本语义发生重大变化。一些增强方法可能不适用于特殊文本，从而导致不生成增强样本。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `sequential` | <class 'bool'> | `False` | whether combine all augmentation methods to a |
| `aug_num` | typing.Annotated[int, Gt(gt=0)] | `1` | number of augmented samples to be generated. If |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If |
| `replace_similar_word` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `replace_homophone_char` | <class 'bool'> | `False` | whether to open the augmentation method |
| `delete_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `swap_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `replace_equivalent_num` | <class 'bool'> | `False` | whether to open the augmentation method |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/nlpcda_zh_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_nlpcda_zh_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)