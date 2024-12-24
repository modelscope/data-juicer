# Post Tuning Tools

For post tuning formats, we mainly consider 4 formats to support [ModelScope-Swift](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md).

- Messages format (Also as ShareGPT format in LLaMA-Factory):

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

- ShareGPT format:

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

- Alpaca format:

```python
{
  "system": "<system>",
  "instruction": "<query-inst>",
  "input": "<query-input>",
  "output": "<response>"
}
```

- Query-Response format:

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

In Data-Juicer, we use the Query-Response format as our intermediate format for post tuning dialog datasets. Thus, Data-Juicer provides several tools to convert datasets in other formats to Query-Response format and vice versa.
