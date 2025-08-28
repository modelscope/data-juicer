# human_preference_annotation_mapper

Operator for human preference annotation using Label Studio.

This operator formats and presents pairs of answers to a prompt for human evaluation. It
uses a default or custom Label Studio configuration to display the prompt and answer
options. The operator processes the annotations to determine the preferred answer,
updating the sample with the chosen and rejected answers. The operator requires specific
keys in the samples for the prompt and answer options. If these keys are missing, it
logs warnings and uses placeholder text. The annotated results are processed to update
the sample with the chosen and rejected answers.

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `label_config_file` | <class 'str'> | `None` |  |
| `answer1_key` | <class 'str'> | `'answer1'` |  |
| `answer2_key` | <class 'str'> | `'answer2'` |  |
| `prompt_key` | <class 'str'> | `'prompt'` |  |
| `chosen_key` | <class 'str'> | `'chosen'` |  |
| `rejected_key` | <class 'str'> | `'rejected'` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/human_preference_annotation_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/annotation/test_human_preference_annotation_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)