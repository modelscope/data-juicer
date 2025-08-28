# query_intent_detection_mapper

Predicts the user's intent label and corresponding score for a given query. The operator
uses a Hugging Face model to classify the intent of the input query. If the query is in
Chinese, it can optionally be translated to English using another Hugging Face
translation model before classification. The predicted intent label and its confidence
score are stored in the meta field with the keys 'query_intent_label' and
'query_intent_score', respectively. If these keys already exist in the meta field, the
operator will skip processing for those samples.

Type 算子类型: **mapper**

Tags 标签: cpu, hf, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'bespin-global/klue-roberta-small-3i4k-intent-classification'` | Huggingface model ID to predict intent label. |
| `zh_to_en_hf_model` | typing.Optional[str] | `'Helsinki-NLP/opus-mt-zh-en'` | Translation model from Chinese to English. |
| `model_params` | typing.Dict | `{}` | model param for hf_model. |
| `zh_to_en_model_params` | typing.Dict | `{}` | model param for zh_to_hf_model. |
| `label_key` | <class 'str'> | `'query_intent_label'` | The key name in the meta field to store the |
| `score_key` | <class 'str'> | `'query_intent_label_score'` | The key name in the meta field to store the |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/query_intent_detection_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_query_intent_detection_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)