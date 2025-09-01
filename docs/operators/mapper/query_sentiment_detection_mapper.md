# query_sentiment_detection_mapper

Predicts user's sentiment label ('negative', 'neutral', 'positive') in a query.

This mapper takes input from the specified query key and outputs the predicted sentiment label and its corresponding score. The results are stored in the Data-Juicer meta field under 'query_sentiment_label' and 'query_sentiment_label_score'. It uses a Hugging Face model for sentiment detection. If a Chinese-to-English translation model is provided, it first translates the query from Chinese to English before performing sentiment analysis.

在查询中预测用户的情绪标签 (“负面” 、 “中性” 、 “正面”)。

此映射器从指定的查询键获取输入，并输出预测的情感标签及其相应的分数。结果存储在 “query_sentiment_label” 和 “query_sentiment_label_score” 下的数据榨汁机元字段中。它使用拥抱人脸模型进行情绪检测。如果提供了中英文翻译模型，则它首先将查询从中文翻译成英文，然后执行情感分析。

Type 算子类型: **mapper**

Tags 标签: cpu, hf, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'` | Huggingface model ID to predict sentiment label. |
| `zh_to_en_hf_model` | typing.Optional[str] | `'Helsinki-NLP/opus-mt-zh-en'` | Translation model from Chinese to English. |
| `model_params` | typing.Dict | `{}` | model param for hf_model. |
| `zh_to_en_model_params` | typing.Dict | `{}` | model param for zh_to_hf_model. |
| `label_key` | <class 'str'> | `'query_sentiment_label'` | The key name in the meta field to store the |
| `score_key` | <class 'str'> | `'query_sentiment_label_score'` | The key name in the meta field to store the |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/query_sentiment_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_query_sentiment_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)