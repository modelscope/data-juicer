[[中文主页]](README_ZH.md) | [[Docs]](#documents) | [[API]](https://modelscope.github.io/data-juicer) |  [[DJ-SORA]](docs/DJ_SORA.md) | [[Awesome List]](docs/awesome_llm_data.md)


# Data-Juicer: A One-Stop Data Processing System for Large Language Models

 <img src="https://img.alicdn.com/imgextra/i3/O1CN017Eq5kf27AlA2NUKef_!!6000000007757-0-tps-1280-720.jpg" width = "640" height = "360" alt="Data-Juicer"/>

![](https://img.shields.io/badge/language-Python-214870.svg)
![](https://img.shields.io/badge/license-Apache--2.0-000000.svg)
[![pypi version](https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad)](https://pypi.org/project/py-data-juicer)
[![Docker version](https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf)](https://hub.docker.com/r/datajuicer/data-juicer)

[![DataModality](https://img.shields.io/badge/DataModality-Text,Image,Audio,Video-brightgreen.svg)](docs/DeveloperGuide_ZH.md)
[![Usage](https://img.shields.io/badge/Usage-Cleaning,Generation,Analysis-FFD21E.svg)](docs/DeveloperGuide_ZH.md)
[![ModelScope- Demos](https://img.shields.io/badge/ModelScope-Demos-4e29ff.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjI0IDEyMS4zMyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxwYXRoIGQ9Im0wIDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtOTkuMTQgNzMuNDloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xNzYuMDkgOTkuMTRoLTI1LjY1djIyLjE5aDQ3Ljg0di00Ny44NGgtMjIuMTl6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTEyNC43OSA0Ny44NGgyNS42NXYyNS42NWgtMjUuNjV6IiBmaWxsPSIjMzZjZmQxIiAvPgoJPHBhdGggZD0ibTAgMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xOTguMjggNDcuODRoMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xOTguMjggMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xNTAuNDQgMHYyMi4xOWgyNS42NXYyNS42NWgyMi4xOXYtNDcuODR6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTczLjQ5IDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiMzNmNmZDEiIC8+Cgk8cGF0aCBkPSJtNDcuODQgMjIuMTloMjUuNjV2LTIyLjE5aC00Ny44NHY0Ny44NGgyMi4xOXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtNDcuODQgNzMuNDloLTIyLjE5djQ3Ljg0aDQ3Ljg0di0yMi4xOWgtMjUuNjV6IiBmaWxsPSIjNjI0YWZmIiAvPgo8L3N2Zz4K)](https://modelscope.cn/studios?name=Data-Jiucer&page=1&sort=latest&type=1)
[![HuggingFace- Demos](https://img.shields.io/badge/🤗HuggingFace-Demos-4e29ff.svg)](https://huggingface.co/spaces?&search=datajuicer)



[![Document_List](https://img.shields.io/badge/Docs-English-blue?logo=Markdown)](#documents)
[![文档列表](https://img.shields.io/badge/文档-中文-blue?logo=Markdown)](README_ZH.md#documents)
[![API Reference](https://img.shields.io/badge/Docs-API_Reference-blue?logo=Markdown)](https://modelscope.github.io/data-juicer/)
[![Paper](http://img.shields.io/badge/cs.LG-arXiv%3A2309.02033-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2309.02033)




Data-Juicer is a one-stop **multimodal** data processing system to make data higher-quality,
juicier, and more digestible for LLMs.


We provide a [playground](http://8.138.149.181/) with a managed JupyterLab. [Try Data-Juicer](http://8.138.149.181/) straight away in your browser! If you find Data-Juicer useful for your research or development, please kindly cite our [work](#references).

[Platform for AI of Alibaba Cloud (PAI)](https://www.aliyun.com/product/bigdata/learn) has cited our work and integrated Data-Juicer into its data processing products. PAI is an AI Native large model and AIGC engineering platform that provides dataset management, computing power management, model tool chain, model development, model training, model deployment, and AI asset management. For documentation on data processing, please refer to: [PAI-Data Processing for Large Models](https://help.aliyun.com/zh/pai/user-guide/components-related-to-data-processing-for-foundation-models/?spm=a2c4g.11186623.0.0.3e9821a69kWdvX).

Data-Juicer is being actively updated and maintained. We will periodically enhance and add more features, data recipes and datasets. 
We welcome you to join us (via issues, PRs, [Slack](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg?spm=a2c22.12281976.0.0.7a8253f30mgpjw)  channel, [DingDing](https://qr.dingtalk.com/action/joingroup?code=v1,k1,YFIXM2leDEk7gJP5aMC95AfYT+Oo/EP/ihnaIEhMyJM=&_dt_no_comment=1&origin=11) group, ...), in promoting data-model co-development along with research and applications of (multimodal) LLMs!

----

## News
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-08-09] We propose Img-Diff, which enhances the performance of multimodal large language models through *contrastive data synthesis*, achieving a score that is 12 points higher than GPT-4V on the [MMVP benchmark](https://tsb0601.github.io/mmvp_blog/). See more details in our [paper](https://arxiv.org/abs/2408.04594), and download the dataset from [huggingface](https://huggingface.co/datasets/datajuicer/Img-Diff) and [modelscope](https://modelscope.cn/datasets/Data-Juicer/Img-Diff).
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-07-24] "Tianchi Better Synth Data Synthesis Competition for Multimodal Large Models" — Our 4th data-centric LLM competition has kicked off! Please visit the competition's [official website](https://tianchi.aliyun.com/competition/entrance/532251) for more information.
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-07-17] We utilized the Data-Juicer [Sandbox Laboratory Suite](https://github.com/modelscope/data-juicer/blob/main/docs/Sandbox.md) to systematically optimize data and models through a co-development workflow between data and models, achieving a new top spot on the [VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) text-to-video leaderboard. The related achievements have been compiled and published in a [paper](http://arxiv.org/abs/2407.11784), and the model has been released on the [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) and [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) platforms.
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-07-12] Our *awesome list of MLLM-Data* has evolved into a systemic [survey](https://arxiv.org/abs/2407.08583) from model-data co-development perspective. Welcome to [explore](docs/awesome_llm_data.md) and contribute!
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-06-01] ModelScope-Sora "Data Directors" creative sprint—Our third data-centric LLM competition has kicked off! Please visit the competition's [official website](https://tianchi.aliyun.com/competition/entrance/532219) for more information.

<details>
<summary> History News:
</summary>>

- [2024-03-07] We release **Data-Juicer [v0.2.0](https://github.com/alibaba/data-juicer/releases/tag/v0.2.0)** now! 
In this new version, we support more features for **multimodal data (including video now)**, and introduce **[DJ-SORA](docs/DJ_SORA.md)** to provide open large-scale, high-quality datasets for SORA-like models.
- [2024-02-20] We have actively maintained an *awesome list of LLM-Data*, welcome to [visit](docs/awesome_llm_data.md) and contribute!
- [2024-02-05] Our paper has been accepted by SIGMOD'24 industrial track!
- [2024-01-10] Discover new horizons in "Data Mixture"—Our second data-centric LLM competition has kicked off! Please visit the competition's [official website](https://tianchi.aliyun.com/competition/entrance/532174) for more information.
- [2024-01-05] We release **Data-Juicer v0.1.3** now!
In this new version, we support **more Python versions** (3.8-3.10), and support **multimodal** dataset [converting](tools/fmt_conversion/multimodal/README.md)/[processing](docs/Operators.md) (Including texts, images, and audios. More modalities will be supported in the future).
Besides, our paper is also updated to [v3](https://arxiv.org/abs/2309.02033).
- [2023-10-13] Our first data-centric LLM competition begins! Please
  visit the competition's official websites, FT-Data Ranker ([1B Track](https://tianchi.aliyun.com/competition/entrance/532157), [7B Track](https://tianchi.aliyun.com/competition/entrance/532158)), for more information.
</details>


<div id="table" align="center"></div>

Table of Contents
=================

- [Data-Juicer:  A One-Stop Data Processing System for Large Language Models](#data-juicer--a-one-stop-data-processing-system-for-large-language-models)
  - [News](#news)
- [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Documentation Index ](#documentation-index-)
  - [Demos](#demos)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [From Source](#from-source)
    - [Using pip](#using-pip)
    - [Using Docker](#using-docker)
    - [Installation check](#installation-check)
  - [Quick Start](#quick-start)
    - [Data Processing](#data-processing)
    - [Distributed Data Processing](#distributed-data-processing)
    - [Data Analysis](#data-analysis)
    - [Data Visualization](#data-visualization)
    - [Build Up Config Files](#build-up-config-files)
    - [Sandbox](#sandbox)
    - [Preprocess Raw Data (Optional)](#preprocess-raw-data-optional)
    - [For Docker Users](#for-docker-users)
  - [Data Recipes](#data-recipes)
  - [License](#license)
  - [Contributing](#contributing)
  - [Acknowledgement](#acknowledgement)
  - [References](#references)


## Features

![Overview](https://img.alicdn.com/imgextra/i4/O1CN01WYQP3Z1JHsaXaQDK6_!!6000000001004-0-tps-3640-1812.jpg)

- **Systematic & Reusable**:
  Empowering users with a systematic library of 80+ core [OPs](docs/Operators.md), 20+ reusable [config recipes](configs), and 20+ feature-rich
  dedicated [toolkits](#documentation), designed to
  function independently of specific multimodal LLM datasets and processing pipelines.

- **Data-in-the-loop & Sandbox**: Supporting one-stop data-model collaborative development, enabling rapid iteration
  through the [sandbox laboratory](docs/Sandbox.md), and providing features such as feedback loops based on data and model,
  visualization, and multidimensional automatic evaluation, so that you can better understand and improve your data and models.
  ![Data-in-the-loop](https://img.alicdn.com/imgextra/i2/O1CN017U7Zz31Y7XtCJ5GOz_!!6000000003012-0-tps-3640-1567.jpg)

- **Towards production environment**: Providing efficient and parallel data processing pipelines (Aliyun-PAI\Ray\Slurm\CUDA\OP Fusion)
  requiring less memory and CPU usage, optimized with automatic fault-toleration.
  ![sys-perf](https://img.alicdn.com/imgextra/i4/O1CN01Sk0q2U1hdRxbnQXFg_!!6000000004300-0-tps-2438-709.jpg)

- **Comprehensive Data Processing Recipes**: Offering tens of [pre-built data
  processing recipes](configs/data_juicer_recipes/README.md) for pre-training, fine-tuning, en, zh, and more scenarios. Validated on
  reference LLaMA and LLaVA models.
  ![exp_llama](https://img.alicdn.com/imgextra/i2/O1CN019WtUPP1uhebnDlPR8_!!6000000006069-2-tps-2530-1005.png)

- **Flexible & Extensible**: Accommodating most types of data formats (e.g., jsonl, parquet, csv, ...) and allowing flexible combinations of OPs. Feel free to [implement your own OPs](docs/DeveloperGuide.md#build-your-own-ops) for customizable data processing.

- **User-Friendly Experience**: Designed for simplicity, with [comprehensive documentation](#documents), [easy start guides](#quick-start) and [demo configs](configs/README.md), and intuitive configuration with simple adding/removing OPs from [existing configs](configs/config_all.yaml).



## Documentation Index <a name="documents"/>

- [Overview](README.md)
- [Operator Zoo](docs/Operators.md)
- [Configs](configs/README.md)
- [Developer Guide](docs/DeveloperGuide.md)
- [API references](https://modelscope.github.io/data-juicer/)
- [KDD-Tutorial](https://modelscope.github.io/data-juicer/_static/tutorial_kdd24.html)
- ["Bad" Data Exhibition](docs/BadDataExhibition.md)
- [Awesome LLM-Data](docs/awesome_llm_data.md)
- Dedicated Toolkits
  - [Quality Classifier](tools/quality_classifier/README.md)
  - [Auto Evaluation](tools/evaluator/README.md)
  - [Preprocess](tools/preprocess/README.md)
  - [Postprocess](tools/postprocess/README.md)
- [DJ-SORA](docs/DJ_SORA.md)
- [Third-parties (LLM Ecosystems)](thirdparty/README.md)


## Demos
- Introduction to Data-Juicer [[ModelScope](https://modelscope.cn/studios/Data-Juicer/overview_scan/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/overview_scan)]
- Data Visualization:
  - Basic Statistics [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_statistics)]
  - Lexical Diversity [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_diversity)]
  - Operator Insight (Single OP) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visualization_op_insight/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_insight)]
  - Operator Effect (Multiple OPs) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_effect)]
- Data Processing:
  - Scientific Literature (e.g. [arXiv](https://info.arxiv.org/help/bulk_data_s3.html)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_sci_data)]
  - Programming Code (e.g. [TheStack](https://huggingface.co/datasets/bigcode/the-stack)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_code_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_code_data)]
  - Chinese Instruction Data (e.g. [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_cft_zh_data)]
- Tool Pool:
  - Dataset Splitting by Language [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language)]
  - Quality Classifier for CommonCrawl [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_quality_classifier)]
  - Auto Evaluation on [HELM](https://github.com/stanford-crfm/helm) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/auto_evaluation_helm)]
  - Data Sampling and Mixture [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_mixture/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_mixture)]
- Data Processing Loop [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_loop)]

## Prerequisites

- Recommend Python>=3.9,<=3.10
- gcc >= 5 (at least C++14 support)

## Installation

### From Source 

- Run the following commands to install the latest basic `data_juicer` version in
  editable mode:
```shell
cd <path_to_data_juicer>
pip install -v -e .
```

- Some OPs rely on some other too large or low-platform-compatibility third-party libraries. You can install optional dependencies as needed:

```shell
cd <path_to_data_juicer>
pip install -v -e .  # install a minimal dependencies, which support the basic functions
pip install -v -e .[tools] # install a subset of tools dependencies
```

The dependency options are listed below:

| Tag              | Description                                                                                  |
|------------------|----------------------------------------------------------------------------------------------|
| `.` or `.[mini]` | Install minimal dependencies for basic Data-Juicer.                                          |
| `.[all]`         | Install all dependencies except sandbox.                                                     |
| `.[sci]`         | Install all dependencies for all OPs.                                                        |
| `.[dist]`        | Install dependencies for distributed data processing. (Experimental)                         |
| `.[dev]`         | Install dependencies for developing the package as contributors.                             |
| `.[tools]`       | Install dependencies for dedicated tools, such as quality classifiers.                       |
| `.[sandbox]`     | Install all dependencies for sandbox.                                                        |

- Install dependencies for specific OPs

With the growth of the number of OPs, the dependencies of all OPs becomes very heavy. Instead of using the command `pip install -v -e .[sci]` to install all dependencies,
we provide two alternative, lighter options:

  - Automatic Minimal Dependency Installation: During the execution of Data-Juicer, minimal dependencies will be automatically installed. This allows for immediate execution, but may potentially lead to dependency conflicts.

  - Manual Minimal Dependency Installation: To manually install minimal dependencies tailored to a specific execution configuration, run the following command:
    ```shell
    # only for installation from source
    python tools/dj_install.py --config path_to_your_data-juicer_config_file

    # use command line tool
    dj-install --config path_to_your_data-juicer_config_file
    ```

### Using pip

- Run the following command to install the latest released `data_juicer` using `pip`:

```shell
pip install py-data-juicer
```

- **Note**:
  - only the basic APIs in `data_juicer` and two basic tools
    (data [processing](#data-processing) and [analysis](#data-analysis)) are available in this way. If you want customizable
    and complete functions, we recommend you install `data_juicer` [from source](#from-source).
  - The release versions from pypi have a certain lag compared to the latest version from source.
    So if you want to follow the latest functions of `data_juicer`, we recommend you install [from source](#from-source).

### Using Docker

- You can
  - either pull our pre-built image from DockerHub:
    ```shell
    docker pull datajuicer/data-juicer:<version_tag>
    ```

  - or run the following command to build the docker image including the
    latest `data-juicer` with provided [Dockerfile](Dockerfile):

    ```shell
    docker build -t datajuicer/data-juicer:<version_tag> .
    ```

  - The format of `<version_tag>` is like `v0.2.0`, which is the same as release version tag.

### Installation check

```python
import data_juicer as dj
print(dj.__version__)
```

### For Video-related Operators
Before using video-related operators, **FFmpeg** should be installed and accessible via the $PATH environment variable.

You can install FFmpeg using package managers(e.g. sudo apt install ffmpeg on Debian/Ubuntu, brew install ffmpeg on OS X) or visit the [official ffmpeg link](https://ffmpeg.org/download.html).

Check if your environment path is set correctly by running the ffmpeg command from the terminal.


<p align="right"><a href="#table">🔼 back to index</a></p>


## Quick Start


### Data Processing

- Run `process_data.py` tool or `dj-process` command line tool with your config as the argument to process
  your dataset.

```shell
# only for installation from source
python tools/process_data.py --config configs/demo/process.yaml

# use command line tool
dj-process --config configs/demo/process.yaml
```

- **Note:** For some operators that involve third-party models or resources which are not stored locally on your computer, it might be slow for the first running because these ops need to download corresponding resources into a directory first.
The default download cache directory is `~/.cache/data_juicer`. Change the cache location by setting the shell environment variable, `DATA_JUICER_CACHE_HOME` to another directory, and you can also change `DATA_JUICER_MODELS_CACHE` or `DATA_JUICER_ASSETS_CACHE` in the same way:

- **Note:** When using operators with third-party models, it's necessary to declare the corresponding `mem_required` in the configuration file (you can refer to the settings in the `config_all.yaml` file). During runtime, Data-Juicer will control the number of processes based on memory availability and the memory requirements of the operator models to achieve better data processing efficiency. When running with CUDA environment, if the mem_required for an operator is not declared correctly, it could potentially lead to a CUDA Out of Memory issue.

```shell
# cache home
export DATA_JUICER_CACHE_HOME="/path/to/another/directory"
# cache models
export DATA_JUICER_MODELS_CACHE="/path/to/another/directory/models"
# cache assets
export DATA_JUICER_ASSETS_CACHE="/path/to/another/directory/assets"
```

#### Flexible Programming Interface
We provide various simple interfaces for users to choose from as follows. 
```python
#... init op & dataset ...

# Chain call style, support single operator or operator list
dataset = dataset.process(op)
dataset = dataset.process([op1, op2])
# Functional programming style for quick integration or script prototype iteration
dataset = op(dataset)
dataset = op.run(dataset)
```


### Distributed Data Processing

We have now implemented multi-machine distributed data processing based on [RAY](https://www.ray.io/). The corresponding demos can be run using the following commands:

```shell
# Run text data processing
python tools/process_data.py --config ./demos/process_on_ray/configs/demo.yaml
# Run video data processing
python tools/process_data.py --config ./demos/process_video_on_ray/configs/demo.yaml
```

- To run data processing across multiple machines, it is necessary to ensure that all distributed nodes can access the corresponding data paths (for example, by mounting the respective data paths on a file-sharing system such as NAS).
- The deduplicator operators for RAY mode are different from the single-machine version, and all those operators are prefixed with `ray`, e.g. `ray_video_deduplicator` and `ray_document_deduplicator`. Those operators also rely on a [Redis](https://redis.io/) instance. So in addition to starting the RAY cluster, you also need to setup your Redis instance in advance and provide `host` and `port` of your Redis instance in configuration.

> Users can also opt not to use RAY and instead split the dataset to run on a cluster with [Slurm](https://slurm.schedmd.com/). In this case, please use the default Data-Juicer without RAY.
> [Aliyun PAI-DLC](https://www.aliyun.com/activity/bigdata/pai-dlc) supports the RAY framework, Slurm framework, etc. Users can directly create RAY jobs and Slurm jobs on the DLC cluster.

### Data Analysis
- Run `analyze_data.py` tool or `dj-analyze` command line tool with your config as the argument to analyze your dataset.

```shell
# only for installation from source
python tools/analyze_data.py --config configs/demo/analyzer.yaml

# use command line tool
dj-analyze --config configs/demo/analyzer.yaml

# you can also use auto mode to avoid writing a recipe. It will analyze a small
# part (e.g. 1000 samples, specified by argument `auto_num`) of your dataset 
# with all Filters that produce stats.
dj-analyze --auto --dataset_path xx.jsonl [--auto_num 1000]
```

- **Note:** Analyzer only compute stats for Filters that produce stats or other OPs that produce tags/categories in meta. So other OPs will be ignored in the analysis process. We use the following registries to decorate OPs:
  - `NON_STATS_FILTERS`: decorate Filters that **DO NOT** produce any stats.
  - `TAGGING_OPS`: decorate OPs that **DO** produce tags/categories in meta field.

### Data Visualization

- Run `app.py` tool to visualize your dataset in your browser.
- **Note**: only available for installation from source.

```shell
streamlit run app.py
```

### Build Up Config Files

- Config files specify some global arguments, and an operator list for the
  data process. You need to set:
  - Global arguments: input/output dataset path, number of workers, etc.
  - Operator list: list operators with their arguments used to process the dataset.
- You can build up your own config files by:
  - ➖：Modify from our example config file [`config_all.yaml`](configs/config_all.yaml) which includes **all** ops and default
    arguments. You just need to **remove** ops that you won't use and refine
    some arguments of ops.
  - ➕：Build up your own config files **from scratch**. You can refer our
    example config file [`config_all.yaml`](configs/config_all.yaml), [op documents](docs/Operators.md), and advanced [Build-Up Guide for developers](docs/DeveloperGuide.md#build-your-own-configs).
  - Besides the yaml files, you also have the flexibility to specify just
    one (of several) parameters on the command line, which will override
    the values in yaml files.

```shell
python xxx.py --config configs/demo/process.yaml --language_id_score_filter.lang=en
```

- The basic config format and definition is shown below.

  ![Basic config example of format and definition](https://img.alicdn.com/imgextra/i1/O1CN01uXgjgj1khWKOigYww_!!6000000004715-0-tps-1745-871.jpg "Basic config file example")

### Sandbox

The data sandbox laboratory (DJ-Sandbox) provides users with the best practices for continuously producing data recipes. It features low overhead, portability, and guidance.

- In the sandbox, users can quickly experiment, iterate, and refine data recipes based on small-scale datasets and models, before scaling up to produce high-quality data to serve large-scale models.
- In addition to the basic data optimization and recipe refinement features offered by Data-Juicer, users can seamlessly use configurable components such as data probe and analysis, model training and evaluation, and data and model feedback-based recipe refinement to form a complete one-stop data-model research and development pipeline.

The sandbox is run using the following commands by default, and for more information and details, please refer to the [sandbox documentation](docs/Sandbox.md).
```shell
python tools/sandbox_starter.py --config configs/demo/sandbox/sandbox.yaml
```

### Preprocess Raw Data (Optional)
- Our formatters support some common input dataset formats for now:
  - Multi-sample in one file: jsonl/json, parquet, csv/tsv, etc.
  - Single-sample in one file: txt, code, docx, pdf, etc.
- However, data from different sources are complicated and diverse. Such as:
  - [Raw arXiv data downloaded from S3](https://info.arxiv.org/help/bulk_data_s3.html) include thousands of tar files and even more gzip files in them, and expected tex files are embedded in the gzip files so they are hard to obtain directly.
  - Some crawled data include different kinds of files (pdf, html, docx, etc.). And extra information like tables, charts, and so on is hard to extract.
- It's impossible to handle all kinds of data in Data-Juicer, issues/PRs are welcome to contribute to process new data types!
- Thus, we provide some **common preprocessing tools** in [`tools/preprocess`](tools/preprocess/) for you to preprocess these data.
  - You are welcome to make your contributions to new preprocessing tools for the community.
  - We **highly recommend** that complicated data can be preprocessed to jsonl or parquet files.

### For Docker Users

- If you build or pull the docker image of `data-juicer`, you can run the commands or tools mentioned above using this docker image.
- Run directly:

```shell
# run the data processing directly
docker run --rm \  # remove container after the processing
  --privileged \
  --shm-size 256g \
  --network host \
  --gpus all \
  --name dj \  # name of the container
  -v <host_data_path>:<image_data_path> \  # mount data or config directory into the container
  -v ~/.cache/:/root/.cache/ \  # mount the cache directory into the container to reuse caches and models (recommended)
  datajuicer/data-juicer:<version_tag> \  # image to run
  dj-process --config /path/to/config.yaml  # similar data processing commands
```

- Or enter into the running container and run commands in editable mode:

```shell
# start the container
docker run -dit \  # run the container in the background
  --privileged \
  --shm-size 256g \
  --network host \
  --gpus all \
  --rm \
  --name dj \
  -v <host_data_path>:<image_data_path> \
  -v ~/.cache/:/root/.cache/ \
  datajuicer/data-juicer:latest /bin/bash

# enter into this container and then you can use data-juicer in editable mode
docker exec -it <container_id> bash
```


<p align="right"><a href="#table">🔼 back to index</a></p>

## Data Recipes
- [Recipes for data process in BLOOM](configs/reproduced_bloom/README.md)
- [Recipes for data process in RedPajama](configs/redpajama/README.md)
- [Refined recipes for pre-training text data](configs/data_juicer_recipes/README.md)
- [Refined recipes for fine-tuning text data](configs/data_juicer_recipes/README.md#before-and-after-refining-for-alpaca-cot-dataset)
- [Refined recipes for pre-training multi-modal data](configs/data_juicer_recipes/README.md#before-and-after-refining-for-multimodal-dataset)



## License
Data-Juicer is released under Apache License 2.0.

## Contributing
We are in a rapidly developing field and greatly welcome contributions of new
features, bug fixes and better documentations. Please refer to
[How-to Guide for Developers](docs/DeveloperGuide.md).

If you have any questions, please join our [discussion groups](README.md).

## Acknowledgement
Data-Juicer is used across various LLM products and research initiatives,
including industrial LLMs from Alibaba Cloud's Tongyi, such as Dianjin for
financial analysis, and Zhiwen for reading assistant, as well as the Alibaba
Cloud's platform for AI (PAI).
We look forward to more of your experience, suggestions and discussions for collaboration!

Data-Juicer thanks and refers to several community projects, such as
[Huggingface-Datasets](https://github.com/huggingface/datasets), [Bloom](https://huggingface.co/bigscience/bloom), [RedPajama](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1), [Pile](https://huggingface.co/datasets/EleutherAI/pile), [Alpaca-Cot](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [DeepSpeed](https://www.deepspeed.ai/), [Arrow](https://github.com/apache/arrow), [Ray](https://github.com/ray-project/ray), [Beam](https://github.com/apache/beam),  [LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness), [HELM](https://github.com/stanford-crfm/helm), ....



## References
If you find our work useful for your research or development, please kindly cite the following [paper](https://arxiv.org/abs/2309.02033).
```
@inproceedings{chen2024datajuicer,
  title={Data-Juicer: A One-Stop Data Processing System for Large Language Models},
  author={Daoyuan Chen and Yilun Huang and Zhijian Ma and Hesen Chen and Xuchen Pan and Ce Ge and Dawei Gao and Yuexiang Xie and Zhaoyang Liu and Jinyang Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
  booktitle={International Conference on Management of Data},
  year={2024}
}
```

<details>
<summary> More related papers from Data-Juicer Team:
</summary>>

- [Data-Juicer Sandbox: A Comprehensive Suite for Multimodal Data-Model Co-development](https://arxiv.org/abs/2407.11784)

- [The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective](https://arxiv.org/abs/2407.08583)

- [ImgDiff: Contrastive Data Synthesis for Vision Large Language Models](https://arxiv.org/abs/2408.04594)

- [Data Mixing Made Efficient: A Bivariate Scaling Law for Language Model Pretraining](https://arxiv.org/abs/2405.14908)

</details>



<p align="right"><a href="#table">🔼 back to index</a></p>
