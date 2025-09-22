# flagged_words_filter

Filter to keep samples with flagged-word ratio in a specified range.

This operator filters out samples based on the ratio of flagged words. It uses a list of flagged words, which can be language-specific or combined from multiple languages. The flagged-word ratio is computed as the number of flagged words divided by the total number of words in the sample. If tokenization is enabled, a Hugging Face tokenizer is used to split the text into words. The operator supports word augmentation for certain languages, which can be configured. The key metric, 'flagged_words_ratio', is cached and reused if already computed. Samples are kept if their flagged-word ratio falls within the specified min and max ratio.

筛选出标记词比率在指定范围内的样本。

该算子根据标记词的比例筛选样本。它使用一个标记词列表，该列表可以是特定语言的或来自多种语言的组合。标记词比率计算为标记词数量除以样本中的总词数。如果启用了分词，则使用 Hugging Face 分词器将文本拆分为单词。该算子支持某些语言的词增强，可以进行配置。关键指标 'flagged_words_ratio' 如果已经计算过则会被缓存并重用。如果样本的标记词比率落在指定的最小和最大比率范围内，则保留该样本。

Type 算子类型: **filter**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `lang` | <class 'str'> | `'en'` | Consider flagged words in what language. If lang == "all", we will adopt the one merged from all the available languages |
| `tokenization` | <class 'bool'> | `False` | Whether to use model to tokenize documents |
| `min_ratio` | <class 'float'> | `0.0` | The min filter ratio in this op. |
| `max_ratio` | <class 'float'> | `0.045` | The max filter ratio in this op. |
| `flagged_words_dir` | <class 'str'> | `'/home/cmgzn/.cache/data_juicer/assets'` | The directory storing the flagged_words file(s) whose name includes "flagged_words" and in json format |
| `use_words_aug` | <class 'bool'> | `False` | Whether to augment words, especially for Chinese and Vietnamese |
| `words_aug_group_sizes` | typing.List[typing.Annotated[int, Gt(gt=0)]] | `[2]` | The group size of words to augment |
| `words_aug_join_char` | <class 'str'> | `''` | The join char between words to augment |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_en_case
```python
FlaggedWordFilter(lang='en', max_ratio=0.045)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is anal cumshot day</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Fuck you doggystyle!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Do you need a cup of coffee?</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 5:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">emoji表情测试下😊，😸31231
</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Do you need a cup of coffee?</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">emoji表情测试下😊，😸31231
</pre></div>

#### ✨ explanation 解释
The FlaggedWordFilter operator filters out samples where the ratio of flagged words to total characters is greater than 0.045. In this case, the first and second samples are removed because they contain a high ratio of flagged English words ('anal', 'cumshot', 'Fuck', 'doggystyle'). The remaining samples do not contain enough flagged content to exceed the specified ratio, so they are kept.算子FlaggedWordFilter会过滤掉标记词比例超过0.045的样本。在该中，第一和第二个样本被移除，因为它们包含了太多的标记英文词('anal', 'cumshot', 'Fuck', 'doggystyle')。剩下的样本不包含足够大量的标记内容以超过指定比例，因此被保留。

### test_zh_case
```python
FlaggedWordFilter(lang='zh', tokenization=True, max_ratio=0.045, use_words_aug=True)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">你是个卖淫女</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">根据算子使用情况增量安装方案确定</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">去除字母、数字、下划线占比过低或过高的代码</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">基于前一步结果，除掉打飞机、三级片等敏感词</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 5:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分，由此过滤低质量文本</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">根据算子使用情况增量安装方案确定</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">去除字母、数字、下划线占比过低或过高的代码</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分，由此过滤低质量文本</pre></div>

#### ✨ explanation 解释
The FlaggedWordFilter operator, with tokenization and word augmentation enabled, filters out Chinese samples where the ratio of flagged words to total tokens is greater than 0.045. Here, the first and fourth samples are removed due to containing sensitive terms ('卖淫女', '打飞机', '三级片'), leading to a flagged word ratio exceeding the threshold. The rest of the samples do not have a high enough flagged word ratio, so they are retained.开启了分词和词汇的FlaggedWordFilter算子会过滤掉标记词比例超过0.045的中文样本。在这里，第一和第四个样本被移除，因为它们包含了敏感词汇(卖淫女, 打飞机, 三级片)，导致标记词比例超过了限值。剩下的样本不存在足够高的标记词比例，因此被保留。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/flagged_words_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_flagged_words_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)