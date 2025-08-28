# in_context_influence_filter

Filter to keep texts based on their in-context influence on a validation set.

This operator calculates the in-context influence of each sample by comparing
perplexities with and without the sample as context. The influence score is computed as
the ratio of these perplexities. If `valid_as_demo` is True, the score is L(A|Q) /
L(A|task_desc, Q_v, A_v, Q). Otherwise, it is L(A_v|Q) / L(A_v|task_desc, Q, A, Q_v).
The operator retains samples whose in-context influence score is within a specified
range. The in-context influence score is stored in the 'in_context_influence' field of
the sample's stats. The validation set must be prepared using the
`prepare_valid_feature` method if not provided during initialization.

Type 算子类型: **filter**

Tags 标签: cpu, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `valid_dataset` | typing.Optional[typing.List[typing.Dict]] | `None` | The dataset to use for validation. |
| `task_desc` | <class 'str'> | `None` | The description of the validation task. |
| `valid_as_demo` | <class 'bool'> | `False` | If true, score =  L(A|Q) / L(A|task_desc, Q_v, A_v, Q); |
| `n_shot` | typing.Optional[int] | `None` | The number of shots in validation. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
### test_sample_as_demo
```python
InContextInfluenceFilter(hf_model=self._hf_model, min_score=1.0, max_score=100.0, query_template='{text}', response_template='{answer}', valid_as_demo=False)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> What is the capital of France?</div><div><strong>A:</strong> The capital of France is Paris.</div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> Explain gravity.</div><div><strong>A:</strong> Gravity is a fundamental force pulling objects toward each other.</div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">What is the capital of France?</pre></div>

#### ✨ explanation 解释
The operator filters the input dataset to keep only those samples whose in-context influence score, calculated as the perplexity ratio of the validation set's answer with and without the sample as context, falls within the specified range [1.0, 100.0]. In this case, the first sample 'What is the capital of France?' is kept because its in-context influence score meets the criteria, while the second sample 'Explain gravity.' is removed for not meeting the criteria.
该算子过滤输入数据集，仅保留那些上下文影响分数（通过计算验证集答案在有无该样本作为上下文情况下的困惑度比率得出）落在指定范围[1.0, 100.0]内的样本。在这种情况下，第一个样本'法国的首都是什么？'被保留是因为其上下文影响分数符合标准，而第二个样本'解释重力。'由于不符合标准而被移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/in_context_influence_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_in_context_influence_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)