# query_topic_detection_mapper

Predicts the topic label and its corresponding score for a given query. The input is
taken from the specified query key. The output, which includes the predicted topic label
and its score, is stored in the 'query_topic_label' and 'query_topic_label_score' fields
of the Data-Juicer meta field. This operator uses a Hugging Face model for topic
classification. If a Chinese to English translation model is provided, it will first
translate the query from Chinese to English before predicting the topic.

- Uses a Hugging Face model for topic classification.
- Optionally translates Chinese queries to English using another Hugging Face
model.
- Stores the predicted topic label in 'query_topic_label'.
- Stores the corresponding score in 'query_topic_label_score'.

Type 算子类型: **mapper**

Tags 标签: cpu, hf, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'dstefa/roberta-base_topic_classification_nyt_news'` | Huggingface model ID to predict topic label. |
| `zh_to_en_hf_model` | typing.Optional[str] | `'Helsinki-NLP/opus-mt-zh-en'` | Translation model from Chinese to English. |
| `model_params` | typing.Dict | `{}` | model param for hf_model. |
| `zh_to_en_model_params` | typing.Dict | `{}` | model param for zh_to_hf_model. |
| `label_key` | <class 'str'> | `'query_topic_label'` | The key name in the meta field to store the |
| `score_key` | <class 'str'> | `'query_topic_label_score'` | The key name in the meta field to store the |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/query_topic_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_query_topic_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)