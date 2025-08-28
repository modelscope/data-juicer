# remove_long_words_mapper

Mapper to remove long words within a specific range.

This operator filters out words in the text that are either shorter than the specified
minimum length or longer than the specified maximum length. Words are first checked with
their original length, and if they do not meet the criteria, they are stripped of
special characters and re-evaluated. The key metric used is the character-based length
of each word. The processed text retains only the words that fall within the defined
length range. This operator processes text in batches for efficiency.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_len` | <class 'int'> | `1` | The min mapper word length in this op, words |
| `max_len` | <class 'int'> | `9223372036854775807` | The max mapper word length in this op, words |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_long_short_words_case
```python
RemoveLongWordsMapper(min_len=3, max_len=15)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper a novel eqeqweqwewqeqwe121e1 method on LLM pretrain.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur la plateforme MT4, manières à ces fonctionnalités sont conçu</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper novel method LLM pretrain.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur plateforme MT4, manières ces fonctionnalités sont conçu</pre></div>

#### ✨ explanation 解释
The operator removes words shorter than 3 characters or longer than 15 characters. In the first sample, 'a' and 'eqeqweqwewqeqwe121e1' are removed because 'a' is too short and 'eqeqweqwewqeqwe121e1' is too long after removing special characters. In the second sample, 'la', 'à', and 'sont' are removed due to being too short.
算子移除长度小于3个字符或大于15个字符的单词。在第一个样本中，'a' 和 'eqeqweqwewqeqwe121e1' 被移除，因为 'a' 太短而 'eqeqweqwewqeqwe121e1' 在去除特殊字符后太长。在第二个样本中，'la'、'à' 和 'sont' 因为太短被移除。

### test_special_words_case
```python
RemoveLongWordsMapper(min_len=3, max_len=15)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed a novel eqeqweqwewqenhq😊😠 method on LLM.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur la plateforme MT4, plusieurs manières d&#x27;accéder0123813976125</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed novel eqeqweqwewqenhq😊😠 method LLM.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur plateforme MT4, plusieurs manières d&#x27;accéder0123813976125</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The Mona Lisa have eyebrows.</pre></div>

#### ✨ explanation 解释
The operator retains or removes words based on their length after stripping special characters. In the first sample, 'a' is removed for being too short, but 'eqeqweqwewqenhq😊😠' remains as it's within the specified range. In the third sample, most of the sentence is stripped out leaving only 'The Mona Lisa have eyebrows.' due to word lengths not meeting the criteria.
算子根据去除特殊字符后的单词长度保留或移除单词。在第一个样本中，'a' 因为太短被移除，但 'eqeqweqwewqenhq😊😠' 保持不变因为它处于指定范围内。在第三个样本中，大部分句子由于单词长度不符合标准而被移除，只留下 'The Mona Lisa have eyebrows.'。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_long_words_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_long_words_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)