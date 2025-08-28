# sentence_split_mapper

Splits text samples into individual sentences based on the specified language.

This operator uses an NLTK-based tokenizer to split the input text into sentences. The
language for the tokenizer is specified during initialization. The original text in each
sample is replaced with a list of sentences. This operator processes samples in batches
for efficiency. Ensure that the `lang` parameter is set to the appropriate language code
(e.g., "en" for English) to achieve accurate sentence splitting.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `lang` | <class 'str'> | `'en'` | split sentence of text in which language. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_en_text
```python
SentenceSplitMapper('en')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota. The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield employs 3,700 people at its plant in Sioux Falls, South Dakota.
The plant slaughters 19,500 pigs a day — 5 percent of U.S. pork.</pre></div>

#### ✨ explanation 解释
The operator splits the English text into sentences, recognizing sentence boundaries and splitting the text accordingly. The input is a single paragraph about Smithfield's plant, and it is split into two sentences, each addressing different aspects of the plant's operations.
算子将英文文本拆分成句子，识别句子边界并相应地拆分文本。输入是一段关于Smithfield工厂的单段文字，被拆分成两个句子，每个句子分别描述了工厂运营的不同方面。

### test_fr_text
```python
SentenceSplitMapper('fr')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield emploie 3,700 personnes dans son usine de Sioux Falls, dans le Dakota du Sud. L&#x27;usine abat 19 500 porcs par jour, soit 5 % du porc américain.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Smithfield emploie 3,700 personnes dans son usine de Sioux Falls, dans le Dakota du Sud.
L&#x27;usine abat 19 500 porcs par jour, soit 5 % du porc américain.</pre></div>

#### ✨ explanation 解释
The operator processes French text, using language-specific rules to identify and split sentences. In this case, the original text, which discusses employment and production at a Smithfield plant, is divided into two sentences, maintaining the original meaning while making the information more structured.
算子处理法语文本，使用特定于语言的规则来识别和拆分句子。在这个例子中，讨论Smithfield工厂就业和生产的原文被分为两个句子，在保持原文意思的同时使信息更加结构化。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/sentence_split_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_sentence_split_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)