# general_field_filter

Filter to keep samples based on a general field filter condition.

The filter condition is a string that can include logical operators (and/or) and chain comparisons. For example: "10 < num <= 30 and text != 'nothing here' and __dj__meta__.a == 3". The condition is evaluated for each sample, and only samples that meet the condition are kept. The result of the filter condition is stored in the sample's stats under the key 'general_field_filter_condition'. If the filter condition is empty or already computed, the sample is not re-evaluated.

基于通用字段过滤条件来保留样本。

过滤条件是一个字符串，可以包含逻辑运算符（and/or）和链式比较。例如："10 < num <= 30 and text != 'nothing here' and __dj__meta__.a == 3"。该条件会对每个样本进行评估，只有满足条件的样本才会被保留。过滤条件的结果存储在样本的 stats 字段中，键名为 'general_field_filter_condition'。如果过滤条件为空或已经计算过，则不会重新评估样本。

Type 算子类型: **filter**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `filter_condition` | <class 'str'> | `''` | The filter condition as a string. It can include logical operators (and/or) and chain comparisons. For example: "10 < num <= 30 and text != 'nothing here' and __dj__meta__.a == 3". |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
### test_simple_comparison
```python
GeneralFieldFilter(filter_condition='10 < num < 20')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">sample1</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>num</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>5</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">sample2</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>num</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>15</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">sample3</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>num</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>25</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">sample2</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>num</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>15</td></tr><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__stats__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>general_field_filter_condition</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>True</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the operator's ability to filter data based on a simple comparison. The operator checks if the 'num' field is between 10 and 20. Only the sample with 'num' equal to 15 meets this condition, so it is kept in the output. The output also includes an additional '__dj__stats__' field indicating that the condition was met.
这个例子展示了算子基于简单比较来过滤数据的能力。算子检查'num'字段是否在10到20之间。只有'num'等于15的样本满足这个条件，因此它被保留在输出中。输出还包括一个额外的'__dj__stats__'字段，表示条件已被满足。

### test_field_missing
```python
GeneralFieldFilter(filter_condition='num <= 5')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">sample1</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>num</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>5</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">sample2</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">sample3</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>num</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>25</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">sample1</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>num</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>5</td></tr><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__stats__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>general_field_filter_condition</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>True</td></tr></table></div></div>

#### ✨ explanation 解释
This example shows how the operator handles missing fields. The filter condition checks if 'num' is less than or equal to 5. If the 'num' field is missing, the sample is automatically excluded. In this case, only the first sample has a 'num' value of 5 and is kept. The output also includes an additional '__dj__stats__' field indicating that the condition was met.
这个例子展示了算子如何处理缺失的字段。过滤条件检查'num'是否小于或等于5。如果'num'字段缺失，样本将自动被排除。在这种情况下，只有第一个样本的'num'值为5并被保留。输出还包括一个额外的'__dj__stats__'字段，表示条件已被满足。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/general_field_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_general_field_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)