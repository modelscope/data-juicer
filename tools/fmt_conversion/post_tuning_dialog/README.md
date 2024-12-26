# Post Tuning Tools

For post tuning formats, we mainly consider 4 formats to support [ModelScope-Swift](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md).

- Swift's Messages format (Very similar to the LLaMA-Factory's ShareGPT format, with different key names):

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

- Swift's ShareGPT format:

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

- Alpaca format (used in the same definition in Swift and LLaMA-Factory):

```python
{
  "system": "<system>",
  "instruction": "<query-inst>",
  "input": "<query-input>",
  "output": "<response>"
}
```

- Swift's Query-Response format:

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

In Data-Juicer, we pre-set fields to align with the last two formats (Alpaca and Query-Response), which serves as our intermediate format for post-tuning dialog datasets. Correspondingly, we provide several tools to convert datasets in other formats to the following DJ format and vice versa.

- DJ default format for post-tuning OPs:

```python
{
  "system": "<system>",
  "instruction": "<query-inst>",
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
