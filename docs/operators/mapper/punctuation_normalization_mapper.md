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
This operator normalizes unicode punctuations to their English equivalents in the text. It replaces characters such as 。 with '.', ， with ',', and other similar substitutions, resulting in a string where all special unicode punctuation is converted to standard English punctuation.
该算子将文本中的Unicode标点符号转换为它们对应的英文标点。例如，将。替换为'.'，将，替换为','等类似替换，最终结果是所有特殊的Unicode标点都转换成了标准的英文标点。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/punctuation_normalization_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_punctuation_normalization_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)