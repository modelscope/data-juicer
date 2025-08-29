# whitespace_normalization_mapper

Normalizes various types of whitespace characters to standard spaces in text samples.

This mapper converts all non-standard whitespace characters, such as tabs and newlines,
to the standard space character (' ', 0x20). It also trims leading and trailing
whitespace from the text. This ensures consistent spacing across all text samples,
improving readability and consistency. The normalization process is based on a
comprehensive list of whitespace characters, which can be found at
https://en.wikipedia.org/wiki/Whitespace_character.

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
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">x 	              　​‌‍⁠￼y</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">x                       y</pre></div>

#### ✨ explanation 解释
This example shows how the operator transforms a text containing various types of whitespace characters, including tabs and different Unicode spaces, into a standard space. The input text has a mix of these special spaces between 'x' and 'y', and after processing, all these are converted to regular spaces, making the text more consistent and readable. The result is 'x' followed by 24 standard spaces, then 'y'.
该示例展示了算子如何将包含各种空格字符（包括制表符和不同的Unicode空格）的文本转换为标准空格。输入文本在'x'和'y'之间混合了这些特殊空格，处理后，所有这些都被转换成了普通空格，使得文本更加一致且易于阅读。结果是'x'后面跟着24个标准空格，然后是'y'。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/whitespace_normalization_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_whitespace_normalization_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)