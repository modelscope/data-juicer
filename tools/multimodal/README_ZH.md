# 多模态工具

这个文件夹包含了一些在使用 Data-Juicer 之前和之后可以用上的多模态数据集处理脚本和工具。

## 数据集格式转换

由于不同多模态数据集和工作之间的数据集格式差异较大，Data-Juicer 提出了一种新颖的多模态数据集中间格式，并为一些流行的多模态工作提供了若干数据集格式转换工具。

这些工具分为两种类型：
- 其他格式到 Data-Juicer 格式的转换：这些工具在 `source_format_to_data_juicer_format` 目录中。它们可以帮助将其他格式的数据集转换为 Data-Juicer 格式的目标数据集。
- Data-Juicer 格式到其他格式的转换：这些工具在 `data_juicer_format_to_target_format` 目录中。它们可以帮助将 Data-Juicer 格式的数据集转换为目标格式的数据集。

目前，Data-Juicer 支持的数据集格式在下面表格中列出。

| 格式        | source_format_to_data_juicer_format | data_juicer_format_to_target_format | 格式参考                                                                                               |
|-----------|-------------------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------|
| 类LLaVA格式  | `llava_to_dj.py`                    | `dj_to_llava.py`                    | [格式描述](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md#dataset-format) |

对于所有工具，您可以运行以下命令来了解它们的详细用法：

```shell
# 例如：llava_to_dj.py
python tools/multimodal/source_format_to_data_juicer_format/llava_to_dj.py --help
```
在使用这些工具之前，您可能需要查看上表中每个格式的参考资料，以更好地了解详细的格式信息，并理解每个工具的参数含义。

### 注意事项
将源数据集转换为 Data-Juicer 格式并再次转换回来后，可能会有一些微小的差异。然而，这些差异几乎不会影响数据集的语义信息。下面我们将详细展示每个支持的源格式中可能存在的这些微小差异。

#### 类LLaVA格式
类LLaVA格式数据集的格式在 [这里](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md#dataset-format) 有具体的定义。尽管它很简单，但在实际场景中，某些样本可能会出现轻微的变体。

这里我们以LLaVA的 [视觉指令微调数据集](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) 为例，展示这些变化如何影响数据集的格式。下表显示了原始数据集和经过若干处理后数据集之间不同样本的数量。原始数据集中有665,298个样本。

| 处理过程                                      | 不同样本数目  |
|-------------------------------------------|---------|
| 1. 运行 `llava_to_dj.py` 和 `dj_to_llava.py` | 113,501 |
| 2. 将源数据集的id字段由整型转为字符串类型                   | 41,361  |
| 3. 将源数据集中对话的所有value字段前后的空格去除              | 40,688  |
| 4. 在转换后的数据集样本中添加 `'model': ''` 字段         | 1       |

值得注意的是，处理过程 2-4 并不会影响数据集中样本对话的语义，因此我们可以认为数据集格式转换工具的转换结果能够对齐源数据集。

最后，只有1个样本不同。如下所示，原因为其对话中包含一些额外的无用字段（"text"，"markdown"）。但是，原始数据集和转换后的数据集之间的"from"和"value"字段是相同的，因此可以认为这个样本也是对齐的。

```json
# 原始对话
[
    {
        "from":"human",
        "value":"sentry self hosted api 504"
    },
    {
        "from":"gpt",
        "text":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\nSlow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n\nServer overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n\nLarge dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n\nAPI endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\nCheck your network connection and ensure that it is stable.\n\nVerify that the server hosting the API is not overloaded with requests.\n\nOptimize the query to retrieve only the necessary data.\n\nTry using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance.",
        "value":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n2. Verify that the server hosting the API is not overloaded with requests.\n3. Optimize the query to retrieve only the necessary data.\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance.",
        "markdown":{
            "type":"answer-markdown",
            "index":1,
            "answer":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n\n2. Verify that the server hosting the API is not overloaded with requests.\n\n3. Optimize the query to retrieve only the necessary data.\n\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance."
        }
    }
]

# 转换后的对话
[
    {
        "from":"human",
        "value":"sentry self hosted api 504"
    },
    {
        "from":"gpt",
        "value":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n2. Verify that the server hosting the API is not overloaded with requests.\n3. Optimize the query to retrieve only the necessary data.\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance."
    }
]
```
