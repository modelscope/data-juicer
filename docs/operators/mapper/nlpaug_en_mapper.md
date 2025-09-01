# nlpaug_en_mapper

Augments English text samples using various methods from the nlpaug library.

This operator applies a series of text augmentation techniques to generate new samples. It supports both word-level and character-level augmentations, such as deleting, swapping, and inserting words or characters. The number of augmented samples can be controlled, and the original samples can be kept or removed. When multiple augmentation methods are enabled, they can be applied sequentially or independently. Sequential application means each sample is augmented by all enabled methods in sequence, while independent application generates multiple augmented samples for each method. We recommend using 1-3 augmentation methods at a time to avoid significant changes in sample semantics.

使用nlpaug库中的各种方法增强英语文本样本。

此运算符应用一系列文本增强技术来生成新样本。它支持单词级和字符级扩充，例如删除、交换和插入单词或字符。可以控制增加的样本的数量，并且可以保留或移除原始样本。当启用多个增强方法时，它们可以顺序地或独立地应用。顺序应用意味着每个样本由所有启用的方法按顺序增强，而独立应用为每个方法生成多个增强样本。我们建议一次使用1-3种增强方法，以避免示例语义发生重大变化。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `sequential` | <class 'bool'> | `False` | whether combine all augmentation methods to a |
| `aug_num` | typing.Annotated[int, Gt(gt=0)] | `1` | number of augmented samples to be generated. If |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If |
| `delete_random_word` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `swap_random_word` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `spelling_error_word` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `split_random_word` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `keyboard_error_char` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `ocr_error_char` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `delete_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `swap_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `insert_random_char` | <class 'bool'> | `False` | whether to open the augmentation method of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/nlpaug_en_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_nlpaug_en_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)