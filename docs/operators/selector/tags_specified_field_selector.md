# tags_specified_field_selector

Selector to filter samples based on the tags of a specified field.

This operator selects samples where the value of the specified field matches one of the
target tags. The field can be multi-level, with levels separated by dots (e.g.,
'level1.level2'). The operator checks if the specified field exists in the dataset and
if the field value is a string, number, or None. If the field value matches any of the
target tags, the sample is kept. The selection is case-sensitive.

- The `field_key` parameter specifies the field to check.
- The `target_tags` parameter is a list of tags to match against the field value.
- If the dataset has fewer than two samples or if `field_key` is empty, the dataset is
returned unchanged.

Type 算子类型: **selector**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `field_key` | <class 'str'> | `''` | Selector based on the specified value |
| `target_tags` | typing.List[str] | `None` | Target tags to be select. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_tag_select
```python
TagsSpecifiedFieldSelector(field_key='meta.sentiment', target_tags=['happy', 'sad'])
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>meta</td><td style='padding:4px 8px;'>{&#x27;sentiment&#x27;: &#x27;happy&#x27;}</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">b</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>meta</td><td style='padding:4px 8px;'>{&#x27;sentiment&#x27;: &#x27;happy&#x27;}</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">c</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>meta</td><td style='padding:4px 8px;'>{&#x27;sentiment&#x27;: &#x27;sad&#x27;}</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">d</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>meta</td><td style='padding:4px 8px;'>{&#x27;sentiment&#x27;: &#x27;angry&#x27;}</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>meta</td><td style='padding:4px 8px;'>{&#x27;sentiment&#x27;: &#x27;happy&#x27;}</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">b</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>meta</td><td style='padding:4px 8px;'>{&#x27;sentiment&#x27;: &#x27;happy&#x27;}</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">c</pre><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap;'>meta</td><td style='padding:4px 8px;'>{&#x27;sentiment&#x27;: &#x27;sad&#x27;}</td></tr></table></div></div>

#### ✨ explanation 解释
The TagsSpecifiedFieldSelector operator filters the dataset by keeping samples where the 'meta.sentiment' field value matches either 'happy' or 'sad'. The sample with 'angry' sentiment is removed because it does not match any of the target tags.
TagsSpecifiedFieldSelector算子通过保留'meta.sentiment'字段值为'happy'或'sad'的样本，来过滤数据集。带有'angry'情绪的样本被移除，因为它与目标标签不匹配。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/selector/tags_specified_field_selector.py)
- [unit test 单元测试](../../../tests/ops/selector/test_tags_specified_field_selector.py)
- [Return operator list 返回算子列表](../../Operators.md)