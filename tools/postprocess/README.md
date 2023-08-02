# Postprocess tools

This folder contains some postprocess scripts for additional processing of your processed dataset using Data-Juicer.

## Usage

### Mix multiple datasets with optional weights

Use `data_mixture.py` to mix multiple datasets.

This script will randomly select samples from every dataset and mix theses samples and export to a new_dataset.


```shell
python tools/postprocess/data_mixture.py        \
    --data_path             <data_path>         \
    --export_path           <export_path>       \
    --export_shard_size     <export_shard_size> \
    --num_proc              <num_proc>

# get help
python tools/postprocess/data_mixture.py  --help
```

- `data_path`: a dataset file or a list of dataset files or a list of both them, optional weights, if not set, 1.0 as default.
- `export_path`: a dataset file name for exporting mixed dataset, support `json` / `jsonl` / `parquet`.
- `export_shard_size`:  dataset file size in Byte. If not set, mixed dataset will be exported into only one file.
- `num_proc`:  process num to load and export datasets.

- e.g., `python tools/postprocess/data_mixture.py  --data_path  <w1> ds.jsonl <w2> ds_dir <w3> ds_file.json`

**Note:** All datasets must have the same meta field, so we can use `datasets` to align their features.
