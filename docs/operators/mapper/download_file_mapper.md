# download_file_mapper

Mapper to download URL files to local files or load them into memory.

This operator downloads files from URLs and can either save them to a specified directory or load the contents directly into memory. It supports downloading multiple files concurrently and can resume downloads if the `resume_download` flag is set. The operator processes nested lists of URLs, flattening them for batch processing and then reconstructing the original structure in the output. If both `save_dir` and `save_field` are not specified, it defaults to saving the content under the key `image_bytes`. The operator logs any failed download attempts and provides error messages for troubleshooting.

映射器将URL文件下载到本地文件或将其加载到内存中。

此运算符从url下载文件，并可以将它们保存到指定的目录或将内容直接加载到内存中。它支持同时下载多个文件，并且如果设置了 “resume_download” 标志，则可以恢复下载。该运算符处理嵌套的url列表，将它们展平以进行批处理，然后在输出中重建原始结构。如果未指定 “save_dir” 和 “save_field”，则默认将内容保存在键 “image_bytes” 下。操作员将记录所有失败的下载尝试，并提供错误消息以进行故障排除。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `download_field` | <class 'str'> | `None` | The filed name to get the url to download. |
| `save_dir` | <class 'str'> | `None` | The directory to save downloaded files. |
| `save_field` | <class 'str'> | `None` | The filed name to save the downloaded file content. |
| `resume_download` | <class 'bool'> | `False` | Whether to resume download. if True, skip the sample if it exists. |
| `timeout` | <class 'int'> | `30` | Timeout for download. |
| `max_concurrent` | <class 'int'> | `10` | Maximum concurrent downloads. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/download_file_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_download_file_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)