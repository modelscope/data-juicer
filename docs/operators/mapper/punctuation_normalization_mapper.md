# punctuation_normalization_mapper

Normalizes unicode punctuations to their English equivalents in text samples.

This operator processes a batch of text samples and replaces any unicode punctuation
with its corresponding English punctuation. The mapping includes common substitutions
like "，" to ",", "。" to ".", and "“" to ". It iterates over each character in the text,
replacing it if it is found in the predefined punctuation map. The result is a set of
text samples with consistent punctuation formatting.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_case

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">,.,&quot;&quot;&quot;&quot;&quot;&quot;&quot;&quot;&quot;&quot;&#x27;::?!();- - . ~&#x27;...-&lt;&gt;[]%-</pre></div>

#### ✨ explanation 解释
This example demonstrates how the PunctuationNormalizationMapper converts various types of Unicode punctuation into their English equivalents. The input text contains a series of different Unicode punctuations, such as 。 for a full stop and ， for a comma. After processing, all these are replaced with their corresponding standard English punctuation marks. This ensures that the text is more consistent and readable in an English context, which can be particularly useful for downstream tasks that expect or require standardized punctuation.
该示例展示了PunctuationNormalizationMapper如何将各种类型的Unicode标点转换为其英文等效标点。输入文本包含一系列不同的Unicode标点符号，如表示句号的。和表示逗号的，。处理后，所有这些都被替换为相应的标准英文标点符号。这确保了文本在英文上下文中更加一致且易于阅读，对于期望或需要标准化标点的下游任务特别有用。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/punctuation_normalization_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_punctuation_normalization_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)