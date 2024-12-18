# 后微调工具

对于 后微调 数据格式，我们主要考虑 4 种格式来覆盖支持 [ModelScope-Swift](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md) 和 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) 。
它们的**核心内容**区域分别为：

- Messages 格式（在 LLaMA-Factory 中其实为 ShareGPT 格式）：

```python
{
  "messages": [
    {
      "role": "system",
      "content": "<system>"
    },
    {
      "role": "user",
      "content": "<query1>"
    },
    {
      "role": "assistant",
      "content": "<response1>"
    },
    {
      "role": "user",
      "content": "<query2>"
    },
    {
      "role": "assistant",
      "content": "<response2>"
    }
  ]
}
```

- ShareGPT 格式：

```python
{
  "system": "<system>",
  "conversation": [
    {
      "human": "<query1>",
      "assistant": "<response1>"
    },
    {
      "human": "<query2>",
      "assistant": "<response2>"
    }
  ]
}
```

- Alpaca 格式：

```python
{
  "system": "<system>",
  "instruction": "<query-inst>",
  "input": "<query-input>",
  "output": "<response>"
}
```

- Query-Response 格式：

```python
{
  "system": "<system>",
  "query": "<query2>",
  "response": "<response2>",
  "history": [
    [
      "<query1>",
      "<response1>"
    ]
  ]
}
```

在 Data-Juicer 中，我们使用 Query-Response 格式作为我们 后微调对话 数据集的中间格式。
因此，Data-Juicer 提供了若干工具讲其他格式的数据集转换为 Query-Response 格式以及反向转换。
