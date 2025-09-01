# extract_tables_from_html_mapper

Extracts tables from HTML content and stores them in a specified field.

This operator processes HTML content to extract tables. It can either retain or remove HTML tags based on the `retain_html_tags` parameter. If `retain_html_tags` is False, it can also include or exclude table headers based on the `include_header` parameter. The extracted tables are stored in the `tables_field_name` field within the sample's metadata. If no tables are found, an empty list is stored. If the tables have already been extracted, the operator will not reprocess the sample.

从HTML内容中提取表并将其存储在指定字段中。

此运算符处理HTML内容以提取表。它可以根据 'retain_html_tags' 参数保留或删除HTML标记。如果 “retain_html_tags” 为False，则它还可以基于 “include_header” 参数包含或排除表头。提取的表存储在示例元数据中的 “tables_field_name” 字段中。如果没有找到表，则存储空列表。如果已经提取了表，则操作员将不重新处理样本。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tables_field_name` | <class 'str'> | `'html_tables'` | Field name to store the extracted tables. |
| `retain_html_tags` | <class 'bool'> | `False` | If True, retains HTML tags in the tables; |
| `include_header` | <class 'bool'> | `True` | If True, includes the table header; |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/extract_tables_from_html_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_extract_tables_from_html_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)