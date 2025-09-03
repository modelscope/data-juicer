# clean_email_mapper

Cleans email addresses from text samples using a regular expression.

This operator removes or replaces email addresses in the text based on a regular expression pattern. By default, it uses a standard pattern to match email addresses, but a custom pattern can be provided. The matched email addresses are replaced with a specified replacement string, which defaults to an empty string. The operation is applied to each text sample in the batch. If no email address is found in a sample, it remains unchanged.

使用正则表达式从文本样本中清理电子邮件地址。

此算子基于正则表达式模式删除或替换文本中的电子邮件地址。默认情况下，它使用标准模式匹配电子邮件地址，但可以提供自定义模式。匹配到的电子邮件地址将被替换为指定的替换字符串，默认为空字符串。该操作应用于批次中的每个文本样本。如果样本中没有找到电子邮件地址，则保持不变。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `pattern` | typing.Optional[str] | `None` | regular expression pattern to search for within text. |
| `repl` | <class 'str'> | `''` | replacement string, default is empty string. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_clean_email
```python
CleanEmailMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&#x27;happy day euqdh@cjqi.com&#x27;, &#x27;请问你是谁dasoidhao@1264fg.45om&#x27;, &#x27;ftp://examplema-nièrdash@hqbchd.ckdhnfes.cds&#x27;, &#x27;👊23da44sh12@46hqb12chd.ckdhnfes.comd.dasd.asd.dc&#x27;]</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&#x27;happy day &#x27;, &#x27;请问你是谁dasoidhao@1264fg.45om&#x27;, &#x27;ftp://examplema-niè&#x27;, &#x27;👊&#x27;]</pre></div>

#### ✨ explanation 解释
This example demonstrates the default behavior of the CleanEmailMapper, which removes email addresses from the text. The operator uses a regular expression to identify and remove any email addresses found in the 'text' field. In the provided samples, emails like 'euqdh@cjqi.com' and 'rdash@hqbchd.ckdhnfes.cds' are removed, leaving only the non-email parts of the text. The sample with no valid email address ('请问你是谁dasoidhao@1264fg.45om') remains unchanged.
这个例子展示了CleanEmailMapper的默认行为，即从文本中移除电子邮件地址。算子使用正则表达式来识别并移除'text'字段中的任何电子邮件地址。在提供的样本中，像'euqdh@cjqi.com'和'rdash@hqbchd.ckdhnfes.cds'这样的电子邮件被移除，只留下文本中的非电子邮件部分。没有有效电子邮件地址的样本（'请问你是谁dasoidhao@1264fg.45om'）保持不变。

### test_replace_email
```python
CleanEmailMapper(repl='<EMAIL>')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&#x27;happy day euqdh@cjqi.com&#x27;, &#x27;请问你是谁dasoidhao@1264fg.45om&#x27;, &#x27;ftp://examplema-nièrdash@hqbchd.ckdhnfes.cds&#x27;, &#x27;👊23da44sh12@46hqb12chd.ckdhnfes.comd.dasd.asd.dc&#x27;]</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&#x27;happy day &lt;EMAIL&gt;&#x27;, &#x27;请问你是谁dasoidhao@1264fg.45om&#x27;, &#x27;ftp://examplema-niè&lt;EMAIL&gt;&#x27;, &#x27;👊&lt;EMAIL&gt;&#x27;]</pre></div>

#### ✨ explanation 解释
In this case, the CleanEmailMapper is configured to replace email addresses with a specific string '<EMAIL>' instead of removing them. The operator identifies email addresses using a regular expression and replaces each found email with the specified replacement string. This way, the original structure of the sentences is maintained, but all email addresses are replaced with '<EMAIL>'. For instance, 'euqdh@cjqi.com' is replaced by '<EMAIL>', and 'rdash@hqbchd.ckdhnfes.cds' is also replaced by the same string. The sample that does not contain a valid email address remains as it is.
在这种情况下，CleanEmailMapper被配置为用特定字符串'<EMAIL>'替换电子邮件地址而不是移除它们。算子使用正则表达式识别电子邮件地址，并将每个找到的电子邮件替换为指定的替换字符串。这样，句子的原始结构得以保留，但所有电子邮件地址都被替换成了'<EMAIL>'。例如，'euqdh@cjqi.com'被替换为'<EMAIL>'，而'rdash@hqbchd.ckdhnfes.cds'也被替换为相同的字符串。不含有效电子邮件地址的样本保持不变。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/clean_email_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_clean_email_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)