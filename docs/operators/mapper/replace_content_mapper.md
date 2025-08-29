# replace_content_mapper

Replaces content in the text that matches a specific regular expression pattern with a
designated replacement string.

This operator processes text by searching for patterns defined in `pattern` and
replacing them with the corresponding `repl` string. If multiple patterns and
replacements are provided, each pattern is replaced by its respective replacement. The
operator supports both single and multiple patterns and replacements. The regular
expressions are compiled with the `re.DOTALL` flag to match across multiple lines. If
the length of the patterns and replacements do not match, a `ValueError` is raised. This
operation is batched, meaning it processes multiple samples at once.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `pattern` | typing.Union[str, typing.List[str], NoneType] | `None` | regular expression pattern(s) to search for within text |
| `repl` | typing.Union[str, typing.List[str]] | `''` | replacement string(s), default is empty string |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_special_char_pattern_text
```python
ReplaceContentMapper(pattern='●■', repl='<SPEC>')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是一个干净的文本。Including Chinese and English.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">◆●■►▼▲▴∆▻▷❖♡□</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">多个●■►▼这样的特殊字符可以►▼▲▴∆吗？</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">未指定的●■☛₨➩►▼▲特殊字符会☻▷❖被删掉吗？？</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是一个干净的文本。Including Chinese and English.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">◆&lt;SPEC&gt;►▼▲▴∆▻▷❖♡□</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">多个&lt;SPEC&gt;►▼这样的特殊字符可以►▼▲▴∆吗？</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">未指定的&lt;SPEC&gt;☛₨➩►▼▲特殊字符会☻▷❖被删掉吗？？</pre></div>

#### ✨ explanation 解释
The operator searches for the special characters '●■' in the text and replaces them with '<SPEC>'. If the special characters are found, they are replaced; otherwise, the text remains unchanged. This example shows how to replace specific symbols in the text, which is useful for cleaning up or anonymizing data.
算子在文本中查找特殊字符'●■'，并将它们替换为'<SPEC>'。如果找到了这些特殊字符，则会进行替换；否则，文本保持不变。这个例子展示了如何替换文本中的特定符号，这对于清理或匿名化数据非常有用。

### test_raw_digit_pattern_text
```python
ReplaceContentMapper(pattern='\\d+(?:,\\d+)*', repl='<DIGIT>')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是一个123。Including 456 and English.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是一个&lt;DIGIT&gt;。Including &lt;DIGIT&gt; and English.</pre></div>

#### ✨ explanation 解释
The operator looks for sequences of digits (numbers) in the text and replaces each sequence with the word '<DIGIT>'. This is helpful for standardizing the representation of numbers within a text, making it easier to process or analyze further. In this case, all numbers are replaced, regardless of their position in the sentence.
算子在文本中查找数字序列，并将每个数字序列替换为单词'<DIGIT>'。这有助于标准化文本内数字的表示方式，使其更容易进一步处理或分析。在这种情况下，无论数字在句子中的位置如何，都会被替换。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/replace_content_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_replace_content_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)