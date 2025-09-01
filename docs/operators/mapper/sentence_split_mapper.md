# sentence_split_mapper

Splits text samples into individual sentences based on the specified language.

This operator uses an NLTK-based tokenizer to split the input text into sentences. The language for the tokenizer is specified during initialization. The original text in each sample is replaced with a list of sentences. This operator processes samples in batches for efficiency. Ensure that the `lang` parameter is set to the appropriate language code (e.g., "en" for English) to achieve accurate sentence splitting.

根据指定的语言将文本样本拆分为单个句子。

此运算符使用基于NLTK的标记器将输入文本拆分为句子。tokenizer的语言是在初始化期间指定的。每个样本中的原始文本被替换为句子列表。该操作员批量处理样品以提高效率。确保将 'lang' 参数设置为适当的语言代码 (例如，“en” 表示英语)，以实现准确的句子拆分。

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
The operator splits the English text into individual sentences, inserting a newline character (\n) between them. The input text contains two sentences: one about the number of people employed and another about the number of pigs slaughtered daily. The output shows these sentences separated by a newline, making it clear that the text has been split into its component sentences.
算子将英文文本拆分成单独的句子，并在它们之间插入换行符（\n）。输入文本包含两个句子：一个是关于雇用人数，另一个是关于每天屠宰的猪的数量。输出显示这些句子被换行符分隔开，清楚地表明文本已被拆分为各个句子。

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
The operator splits the French text into individual sentences, inserting a newline character (\n) between them. The input text contains two sentences: one about the number of people employed and another about the number of pigs slaughtered daily. The output shows these sentences separated by a newline, making it clear that the text has been split into its component sentences.
算子将法语文本拆分成单独的句子，并在它们之间插入换行符（\n）。输入文本包含两个句子：一个是关于雇用人数，另一个是关于每天屠宰的猪的数量。输出显示这些句子被换行符分隔开，清楚地表明文本已被拆分为各个句子。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/sentence_split_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_sentence_split_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)