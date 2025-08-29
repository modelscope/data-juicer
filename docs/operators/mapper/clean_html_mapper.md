# clean_html_mapper

Cleans HTML code from text samples, converting HTML to plain text.

This operator processes text samples by removing HTML tags and converting HTML elements
to a more readable format. Specifically, it replaces `<li>` and `<ol>` tags with newline
and bullet points. The Selectolax HTML parser is used to extract the text content from
the HTML. This operation is performed in a batched manner, making it efficient for large
datasets.

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_complete_html_text

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;header&gt;&lt;nav&gt;&lt;ul&gt;&lt;tile&gt;测试&lt;/title&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Home&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;About&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Services&lt;/a&gt;&lt;/li&gt;&lt;li&gt;&lt;a href=&quot;#&quot;&gt;Contact&lt;/a&gt;&lt;/li&gt;&lt;/ul&gt;&lt;/nav&gt;&lt;/header&gt;&lt;main&gt;&lt;h1&gt;Welcome to My Website&lt;/h1&gt;&lt;p&gt;Lorem ipsum dolor sit amet, consectetur adipiscing elit.&lt;button&gt;Learn More&lt;/button&gt;&lt;/main&gt;&lt;footer&gt;&lt;p&gt;&amp;copy; 2021 My Website. All Rights Reserved.&lt;/p&gt;&lt;/footer&gt;</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">测试
*Home
*About
*Services
*ContactWelcome to My WebsiteLorem ipsum dolor sit amet, consectetur adipiscing elit.Learn More© 2021 My Website. All Rights Reserved.</pre></div>

#### ✨ explanation 解释
This example demonstrates how the operator cleans a complete HTML text, removing all HTML tags and preserving the text content. The <li> tags are replaced with bullet points to make the list items more readable. The output is a plain text version of the original HTML, which is easier to read and process.
这个例子展示了算子如何清理完整的HTML文本，移除所有的HTML标签并保留文本内容。<li>标签被替换为项目符号，使列表项更易读。输出是原始HTML的纯文本版本，更易于阅读和处理。

### test_no_html_text

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This is a test</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是个测试</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">12345678</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">This is a test</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">这是个测试</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">12345678</pre></div>

#### ✨ explanation 解释
In this example, the operator processes texts that do not contain any HTML tags. As there are no HTML elements to clean, the input and output texts remain the same. This shows that the operator can handle plain text without making any changes, ensuring that non-HTML content is left untouched.
在这个例子中，算子处理不包含任何HTML标签的文本。由于没有需要清理的HTML元素，输入和输出文本保持不变。这表明算子可以处理纯文本而不做任何更改，确保非HTML内容不会受到影响。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/clean_html_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_clean_html_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)