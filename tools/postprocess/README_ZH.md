# Postprocess tools

此文件夹包含一些后处理脚本，用于对 Data-Juicer 处理后的数据集进行进一步处理。

## 用法

### 将多个数据集以可选的权重混合

使用 [data_mixture.py](data_mixture.py) 将多个数据集混合。

该脚本将从每个数据集中随机选择样本并混合这些样本并导出到新的数据集。

```shell
python tools/postprocess/data_mixture.py        \
    --data_path             <data_path>         \
    --export_path           <export_path>       \
    --export_shard_size     <export_shard_size> \
    --num_proc              <num_proc>

# get help
python tools/postprocess/data_mixture.py  --help
```

- `data_path`: 数据集文件或数据集文件列表或两者的列表。可附加可选权重，权重未设置时默认值为 1.0。
- `export_path`: 用于导出混合数据集的数据集文件名，支持 `json` / `jsonl` / `parquet` 格式。
- `export_shard_size`: 数据集文件大小（以字节为单位）。 如果未设置，混合数据集将仅导出到一个文件中。
- `num_proc`:  加载以及导出数据集使用的进程数量

- 例，`python tools/postprocess/data_mixture.py  --data_path  <w1> ds.jsonl <w2> ds_dir <w3> ds_file.json`

**注意事项:** 所有数据集必须具有相同的元字段，从而可以使用 [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) 来对齐它们的特征。
