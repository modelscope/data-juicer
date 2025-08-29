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
This example demonstrates the basic functionality of the RemoveLongWordsMapper operator, where it removes words that are either too short (less than 3 characters) or too long (more than 15 characters). In the first sentence, 'a' is removed because it's too short, and 'eqeqweqwewqeqwe121e1' is removed for being too long. Similarly, in the second sentence, 'la', 'à', and 'sont' are removed for being too short. The output data shows the text after these words have been filtered out.
这个例子展示了RemoveLongWordsMapper算子的基本功能，它会移除太短（少于3个字符）或太长（超过15个字符）的单词。在第一句话中，'a'因为太短被移除，而'eqeqweqwewqeqwe121e1'因为太长被移除。同样，在第二句话中，'la'、'à'和'sont'由于太短也被移除。输出数据显示了这些单词被过滤后的文本。

### test_special_words_case
```python
RemoveLongWordsMapper(min_len=3, max_len=15)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed a novel eqeqweqwewqenhq😊😠 method on LLM.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur la plateforme MT4, plusieurs manières d&#x27;accéder0123813976125</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This paper proposed novel eqeqweqwewqenhq😊😠 method LLM.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur plateforme MT4, plusieurs manières d&#x27;accéder0123813976125</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The Mona Lisa have eyebrows.</pre></div>

#### ✨ explanation 解释
This example covers a more complex scenario, showing how the operator handles special characters and non-English languages. In the first sentence, the word 'eqeqweqwewqenhq😊😠' is kept despite its length because it contains special characters which are not counted towards the length limit. In the third sentence, the original text has some encoding issues, but the operator still correctly processes it by removing the word 'doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t' as it exceeds the maximum length. This case illustrates the robustness of the operator in dealing with special cases and different character sets.
这个例子覆盖了一个更复杂的场景，展示了该算子如何处理特殊字符和非英语语言。在第一句话中，尽管单词'eqeqweqwewqenhq😊😠'很长，但由于其中包含不计入长度限制的特殊字符，因此被保留。在第三句话中，原文存在一些编码问题，但算子仍然正确地处理了它，通过移除超过最大长度的单词'doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t'。这个案例说明了该算子在处理特殊情况和不同字符集时的鲁棒性。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/remove_long_words_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_remove_long_words_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)