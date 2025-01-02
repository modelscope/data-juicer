# Distributed Data Processing in Data-Juicer

In Data-Juicer, we implement distributed data processing based on the famous [Ray](https://github.com/ray-project/ray) framework.
Based on Ray, we optimize the strategy of tuning the number of split blocks of the input dataset in Ray,
and contributed for streaming reading of json files to Ray and Apache Arrow.
Then, we scale distributed data processing of Data-Juicer based on Ray and our patches up to datasets containing billions of samples on tens of thousands of CPU cores for ultimate efficiency.
We also equipped with a MinHash-based deduplication operator based on Ray, which could deduplicate TB-sized datasets on thousand of CPU cores in 3 hours.

For more details, please refer to our Data-Juicer 2.0 paper.

## Functional Optimizations for Ray

### Subset splitting

When there are tens of thousands of nodes but with only a few dataset files, Ray would split the dataset files according to the available resources and distribute the blocks of the dataset to all nodes, which brings a huge network communication cost and decreases the CPU utilization of each node.

Thus, we split the original dataset into smaller 128MB files in advance automatically according to the dataset size and the number of distributed nodes, trying to adapt the features of Arrow and Ray for better performance.
This approach reduces location and reprocessing costs associated with fault tolerance and helps mitigate network exchange overheads, especially beneficial in contexts involving large-scale multimodal data, as well as in scenarios that require handling global objects of Ray Actor in distributed modes.

### Streaming Reading of Json Files

We offer a streaming loading interface, addressing the current lack of native support in the Arrow framework underlying Hugging Face and Ray Datasets for streaming JSON data.
We developed an in-house patch for Apache Arrow ([PR](https://github.com/apache/arrow/pull/45084)) to alleviate Out-of-Memory (OOM) issues.

## Distributed Dataset Processing

Based on the two optimizations above, we conduct experiments on datasets with billions of samples.
We prepare a 560k-sample multimodal dataset and expand it by different factors to get datasets with different sizes. 
The experimental results are shown in the figure below, which demonstrates the scalability.
And our optimizations for Ray offers 2x∼3x speedups in our experiments.

![Overview](https://img.alicdn.com/imgextra/i3/O1CN01JV8wcC1oxn0G2xnBT_!!6000000005292-0-tps-1328-1742.jpg)

## Distributed Deduplication on Large-Scale Datasets

We conduct MinHash-based RayDeduplicator on datasets sized at 200GB, 1TB, and 5TB, using CPU counts ranging from 640 to 1280 cores.
As the table below shows, when the data size increases by 5x, the processing time increases by 4.02x∼5.62x.
When the number of CPU cores doubles, the processing time decreases to 58.9%∼67.1% of the original time.


| # CPU   | 200GB Time | 1TB Time  | 5TB Time   |
|---------|------------|-----------|------------|
| 4 * 160 | 11.13 min  | 50.83 min | 285.43 min |
| 8 * 160 | 7.47 min   | 30.08 min | 168.10 min |

## Examples of Data Processing based on Ray

### Simple Demo of Data Processing Based on Ray Using Data-Juicer OPs

We already prepare a simple demo in the directory `demos/process_on_ray/`, where we put a config file and two test datasets.
```text
demos/process_on_ray/
├── configs
│   └── demo.yaml
└── data
    ├── demo-dataset.json
    └── demo-dataset.jsonl
```

We already set the executor type to "ray" and set an auto ray address in the config file.
```yaml
...
dataset_path: './demos/process_on_ray/data/demo-dataset.jsonl'
export_path: './outputs/demo/demo-processed'

executor_type: 'ray'  # set the executor type to "ray"
ray_address: 'auto'  # set an auto ray address.
...
```

Before running, we need to install Data-Juicer and its `dist` requirements:

```shell
pip install -v -e .  # installing the minimal requirements of Data-Juicer
pip install -v -e ".[dist]"  # including dependencies on ray and other distributed libs
```

Then we need to start a Ray cluster:

```shell
ray start --head  # start a local cluster as the head node
```

And we can run this demo with the `dj-process` tool:

```shell
# run the tool from source
python tools/process_data.py --config demos/process_on_ray/configs/demo.yaml

# use command line tool
dj-process --config demos/process_on_ray/configs/demo.yaml
```

Data-Juicer will process the demo dataset with the demo config file and export the result datasets in the directory specified by the `export_path` argument in the config file.