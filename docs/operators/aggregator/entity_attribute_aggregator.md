# entity_attribute_aggregator

Summarizes a given attribute of an entity from a set of documents.

The operator extracts and summarizes the specified attribute of a given entity from the provided documents. It uses a system prompt, example prompt, and input template to generate the summary. The output is formatted as a markdown-style summary with the entity and attribute clearly labeled. The summary is limited to a specified number of words (default is 100). The operator uses a Hugging Face tokenizer to handle token limits and splits documents if necessary. If the input key or required fields are missing, the operator logs a warning and returns the sample unchanged. The summary is stored in the batch metadata under the specified output key. The system prompt, input template, example prompt, and output pattern can be customized.

从一组文档中提取并总结给定实体的特定属性。

该算子从提供的文档中提取并总结给定实体的指定属性。它使用系统提示、示例提示和输入模板生成摘要。输出格式为 markdown 风格的摘要，其中实体和属性清晰标注。摘要限制在指定的单词数内（默认为 100 个单词）。该算子使用 Hugging Face 分词器来处理 token 限制，并在必要时分割文档。如果缺少输入键或必填字段，该算子会记录警告并返回未更改的样本。摘要存储在批处理元数据中的指定输出键下。系统提示、输入模板、示例提示和输出模式可以自定义。

Type 算子类型: **aggregator**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `entity` | <class 'str'> | `None` | The given entity. |
| `attribute` | <class 'str'> | `None` | The given attribute. |
| `input_key` | <class 'str'> | `'event_description'` | The input key in the meta field of the samples. |
| `output_key` | <class 'str'> | `'entity_attribute'` | The output key in the aggregation field of the |
| `word_limit` | typing.Annotated[int, Gt(gt=0)] | `100` | Prompt the output length. |
| `max_token_num` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The max token num of the total tokens of the |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt_template` | typing.Optional[str] | `None` | The system prompt template. |
| `example_prompt` | typing.Optional[str] | `None` | The example part in the system prompt. |
| `input_template` | typing.Optional[str] | `None` | The input template. |
| `output_pattern_template` | typing.Optional[str] | `None` | The output template. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_default_aggregator
```python
EntityAttributeAggregator(api_model='qwen2.5-72b-instruct', entity='李莲花', attribute='主要经历')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap; font-weight:bold;' colspan='2'>__dj__meta__</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap; font-weight:bold;' colspan='2'>__dj__meta__</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。</td></tr><tr><td style='padding:4px 8px; color:#555; white-space:nowrap; font-weight:bold;' colspan='2'>__dj__batch_meta__</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 20px;'>entity_attribute</td><td style='padding:2px 8px; padding-left: 20px;'>李莲花原名李相夷，十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主。后因与魔教金鸳盟盟主笛飞声激战中毒重伤，船只爆炸沉海。十年后，他在寒酸的莲花楼内醒来，以神医身份在小镇上行医，生活贫困。</td></tr></table></div></div>

#### ✨ explanation 解释
This example demonstrates the default behavior of the operator, which summarizes the main experiences of the character '李莲花' from a set of documents. The output is a summary that includes key events in the life of '李莲花', such as his early achievements and later life as a doctor. The summary is stored under the 'entity_attribute' key in the batch metadata.
此示例展示了算子的默认行为，从一组文档中总结角色'李莲花'的主要经历。输出是一个摘要，包括'李莲花'生活中的关键事件，如他早期的成就和后来作为医生的生活。该摘要存储在批次元数据的'entity_attribute'键下。

### test_word_limit_num
```python
EntityAttributeAggregator(api_model='qwen2.5-72b-instruct', entity='李莲花', attribute='身份背景', word_limit=20)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap; font-weight:bold;' colspan='2'>__dj__meta__</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin-top:6px;'><table style='border-collapse:collapse; margin-top:6px;'><tr><td style='padding:4px 8px; color:#555; white-space:nowrap; font-weight:bold;' colspan='2'>__dj__meta__</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 40px;'>event_description</td><td style='padding:2px 8px; padding-left: 40px;'>小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。</td></tr><tr><td style='padding:4px 8px; color:#555; white-space:nowrap; font-weight:bold;' colspan='2'>__dj__batch_meta__</td></tr><tr><td style='padding:2px 8px; color:#777; white-space:nowrap; padding-left: 20px;'>entity_attribute</td><td style='padding:2px 8px; padding-left: 20px;'>原名李相夷，曾是天下第一高手、武林盟主，后化名李莲花，以神医身份隐居。</td></tr></table></div></div>

#### ✨ explanation 解释
This example shows how to limit the number of words in the summary. The operator generates a concise summary of the character '李莲花' with a word limit of 20. The output is a brief summary that captures the essence of '李莲花'’s background and experiences. The summary is stored under the 'entity_attribute' key in the batch metadata.
此示例展示了如何限制摘要中的字数。算子生成一个关于角色'李莲花'的简短摘要，字数限制为20。输出是一个简洁的摘要，概括了'李莲花'的背景和经历。该摘要存储在批次元数据的'entity_attribute'键下。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/aggregator/entity_attribute_aggregator.py)
- [unit test 单元测试](../../../tests/ops/aggregator/test_entity_attribute_aggregator.py)
- [Return operator list 返回算子列表](../../Operators.md)