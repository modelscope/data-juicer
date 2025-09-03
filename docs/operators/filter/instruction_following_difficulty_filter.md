# instruction_following_difficulty_filter

Filter to keep texts based on their instruction following difficulty (IFD) score.

This operator computes the IFD score for each text sample, which is the ratio of the loss with the query to the loss without the query. The IFD score is used to determine the difficulty of following an instruction. Samples are kept if their IFD score falls within a specified range. The IFD score is cached in the 'ifd_score' field of the sample's stats. This operator uses a Hugging Face tokenizer and model to compute the losses.

根据指令跟随难度（IFD）分数筛选保留文本。

该算子计算每个文本样本的IFD分数，即带查询的损失与不带查询的损失之比。IFD分数用于确定遵循指令的难度。如果样本的IFD分数落在指定范围内，则保留该样本。IFD分数缓存在样本的stats字段中的'ifd_score'字段中。该算子使用Hugging Face的分词器和模型来计算损失。

Type 算子类型: **filter**

Tags 标签: cpu, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'Qwen/Qwen2.5-0.5B'` | huggingface embedding model name. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the API model. |
| `min_score` | <class 'float'> | `1.0` | Minimum perplexity score. |
| `max_score` | <class 'float'> | `100.0` | Maximum perplexity score. |
| `query_template` | typing.Optional[str] | `None` | Template for building the query string. |
| `response_template` | typing.Optional[str] | `None` | Template for building the response string. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_rft_data
```python
InstructionFollowingDifficultyFilter(hf_model='Qwen/Qwen2.5-0.5B', min_score=0.2, max_score=0.9, query_template='Question: {text}', response_template='Answer: {answer}')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Explain gravity.</pre></div><div><strong>A:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Gravity is a fundamental force pulling objects toward each other.</pre></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">What is the capital of France?</pre></div><div><strong>A:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The capital of France is Paris.</pre></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">How does chocolate taste?</pre></div><div><strong>A:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The capital of France is Paris.</pre></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Explain gravity.</pre></div>

#### ✨ explanation 解释
The operator filters text samples based on their Instruction Following Difficulty (IFD) score, which is the ratio of loss with the query to the loss without the query. Only samples with an IFD score between 0.2 and 0.9 are kept. In this case, only the sample 'Explain gravity.' meets the criteria, while others are filtered out due to their IFD scores falling outside the specified range.
算子根据每个文本样本的指令跟随难度(IFD)分数来过滤数据，该分数是带有查询条件下的损失与无查询条件下的损失之比。只有IFD分数在0.2到0.9之间的样本被保留。在这个例子中，仅'Explain gravity.'这个样本符合标准，而其他样本由于其IFD分数不在指定范围内而被过滤掉。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/instruction_following_difficulty_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_instruction_following_difficulty_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)