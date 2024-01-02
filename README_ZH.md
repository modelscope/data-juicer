[**English**](README.md) | 中文

# Data-Juicer: 为大语言模型提供更高质量、更丰富、更易“消化”的数据

![Data-Juicer](https://img.alicdn.com/imgextra/i3/O1CN017Eq5kf27AlA2NUKef_!!6000000007757-0-tps-1280-720.jpg "Data-Juicer")

[![Paper](http://img.shields.io/badge/cs.LG-arXiv%3A2309.02033-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2309.02033)
![](https://img.shields.io/badge/language-Python-214870.svg)
![](https://img.shields.io/badge/license-Apache--2.0-000000.svg)
[![Contributing](https://img.shields.io/badge/Contribution-welcome-brightgreen.svg)](docs/DeveloperGuide_ZH.md)

[![pypi version](https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad)](https://pypi.org/project/py-data-juicer)
[![Docker version](https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf)](https://hub.docker.com/r/datajuicer/data-juicer)
[![Document_List](https://img.shields.io/badge/Docs-English-blue?logo=Markdown)](README.md#documentation)
[![文档列表](https://img.shields.io/badge/文档-中文-blue?logo=Markdown)](README_ZH.md#documentation)
[![API Reference](https://img.shields.io/badge/Docs-API_Reference-blue?logo=Markdown)](https://alibaba.github.io/data-juicer/)

[![ModelScope-10+ Demos](https://img.shields.io/badge/ModelScope-10+_Demos-4e29ff.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjI0IDEyMS4zMyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxwYXRoIGQ9Im0wIDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtOTkuMTQgNzMuNDloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xNzYuMDkgOTkuMTRoLTI1LjY1djIyLjE5aDQ3Ljg0di00Ny44NGgtMjIuMTl6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTEyNC43OSA0Ny44NGgyNS42NXYyNS42NWgtMjUuNjV6IiBmaWxsPSIjMzZjZmQxIiAvPgoJPHBhdGggZD0ibTAgMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xOTguMjggNDcuODRoMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xOTguMjggMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xNTAuNDQgMHYyMi4xOWgyNS42NXYyNS42NWgyMi4xOXYtNDcuODR6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTczLjQ5IDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiMzNmNmZDEiIC8+Cgk8cGF0aCBkPSJtNDcuODQgMjIuMTloMjUuNjV2LTIyLjE5aC00Ny44NHY0Ny44NGgyMi4xOXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtNDcuODQgNzMuNDloLTIyLjE5djQ3Ljg0aDQ3Ljg0di0yMi4xOWgtMjUuNjV6IiBmaWxsPSIjNjI0YWZmIiAvPgo8L3N2Zz4K)](https://modelscope.cn/studios?name=Data-Jiucer&page=1&sort=latest&type=1)
[![ModelScope-20+_Refined_Datasets](https://img.shields.io/badge/ModelScope-20+_Refined_Datasets-4e29ff.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjI0IDEyMS4zMyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxwYXRoIGQ9Im0wIDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtOTkuMTQgNzMuNDloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xNzYuMDkgOTkuMTRoLTI1LjY1djIyLjE5aDQ3Ljg0di00Ny44NGgtMjIuMTl6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTEyNC43OSA0Ny44NGgyNS42NXYyNS42NWgtMjUuNjV6IiBmaWxsPSIjMzZjZmQxIiAvPgoJPHBhdGggZD0ibTAgMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xOTguMjggNDcuODRoMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xOTguMjggMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xNTAuNDQgMHYyMi4xOWgyNS42NXYyNS42NWgyMi4xOXYtNDcuODR6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTczLjQ5IDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiMzNmNmZDEiIC8+Cgk8cGF0aCBkPSJtNDcuODQgMjIuMTloMjUuNjV2LTIyLjE5aC00Ny44NHY0Ny44NGgyMi4xOXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtNDcuODQgNzMuNDloLTIyLjE5djQ3Ljg0aDQ3Ljg0di0yMi4xOWgtMjUuNjV6IiBmaWxsPSIjNjI0YWZmIiAvPgo8L3N2Zz4K)](https://modelscope.cn/datasets?organization=Data-Juicer&page=1)
[![ModelScope-Reference_Models](https://img.shields.io/badge/ModelScope-Reference_Models-4e29ff.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjI0IDEyMS4zMyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxwYXRoIGQ9Im0wIDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtOTkuMTQgNzMuNDloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xNzYuMDkgOTkuMTRoLTI1LjY1djIyLjE5aDQ3Ljg0di00Ny44NGgtMjIuMTl6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTEyNC43OSA0Ny44NGgyNS42NXYyNS42NWgtMjUuNjV6IiBmaWxsPSIjMzZjZmQxIiAvPgoJPHBhdGggZD0ibTAgMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xOTguMjggNDcuODRoMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xOTguMjggMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xNTAuNDQgMHYyMi4xOWgyNS42NXYyNS42NWgyMi4xOXYtNDcuODR6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTczLjQ5IDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiMzNmNmZDEiIC8+Cgk8cGF0aCBkPSJtNDcuODQgMjIuMTloMjUuNjV2LTIyLjE5aC00Ny44NHY0Ny44NGgyMi4xOXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtNDcuODQgNzMuNDloLTIyLjE5djQ3Ljg0aDQ3Ljg0di0yMi4xOWgtMjUuNjV6IiBmaWxsPSIjNjI0YWZmIiAvPgo8L3N2Zz4K)](https://modelscope.cn/models?organization=Data-Juicer&page=1)

[![HuggingFace-10+ Demos](https://img.shields.io/badge/🤗HuggingFace-10+_Demos-FFD21E.svg)](https://huggingface.co/spaces?&search=datajuicer)
[![HuggingFace-20+_Refined_Datasets](https://img.shields.io/badge/🤗HuggingFace-20+_Refined_Datasets-FFD21E.svg)](https://huggingface.co/datasets?&search=datajuicer)
[![HuggingFace-Reference_Models](https://img.shields.io/badge/🤗HuggingFace-Reference_Models-FFD21E.svg)](https://huggingface.co/models?&search=datajuicer)

[![QualityClassifier](https://img.shields.io/badge/Tools-Quality_Classifier-saddlebrown?logo=Markdown)](tools/quality_classifier/README_ZH.md)
[![AutoEvaluation](https://img.shields.io/badge/Tools-Auto_Evaluation-saddlebrown?logo=Markdown)](tools/evaluator/README_ZH.md)

Data-Juicer 是一个一站式数据处理系统，旨在为大语言模型 (LLM) 提供更高质量、更丰富、更易“消化”的数据。
本项目在积极更新和维护中，我们将定期强化和新增更多的功能和数据菜谱。欢迎您加入我们推进 LLM 数据的开发和研究工作！

如果Data-Juicer对您的研发有帮助，请引用我们的[工作](#参考文献) 。


----

## 新消息
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2023-10-13] 我们的第一届以数据为中心的 LLM 竞赛开始了！
  请访问大赛官网，**FT-Data Ranker**（[1B赛道](https://tianchi.aliyun.com/competition/entrance/532157) 、[7B赛道](https://tianchi.aliyun.com/competition/entrance/532158) ) ，了解更多信息。

- [2023-10-8] 我们的论文更新至第二版，并发布了对应的Data-Juicer v0.1.2版本！

目录
===

* [Data-Juicer: 为大语言模型提供更高质量、更丰富、更易“消化”的数据](#data-juicer-为大语言模型提供更高质量更丰富更易消化的数据)
* [目录](#目录)
  * [特点](#特点)
  * [前置条件](#前置条件)
  * [安装](#安装)
    * [从源码安装](#从源码安装)
    * [使用 pip 安装](#使用-pip-安装)
    * [使用 Docker 安装](#使用-docker-安装)
    * [安装校验](#安装校验)
  * [快速上手](#快速上手)
    * [数据处理](#数据处理)
    * [数据分析](#数据分析)
    * [数据可视化](#数据可视化)
    * [构建配置文件](#构建配置文件)
    * [预处理原始数据（可选）](#预处理原始数据可选)
    * [对于 Docker 用户](#对于-docker-用户)
  * [Documentation | 文档](#documentation)
  * [数据处理菜谱](#数据处理菜谱)
  * [演示样例](#演示样例)
  * [开源协议](#开源协议)
  * [贡献](#贡献)
  * [致谢](#致谢)
  * [参考文献](#参考文献)

## 特点

![Overview](https://img.alicdn.com/imgextra/i2/O1CN01IMPeD11xYRUYLmXKO_!!6000000006455-2-tps-3620-1604.png)

* **系统化 & 可复用**：为用户提供系统化且可复用的20+[配置菜谱](configs/README_ZH.md)，50+核心[算子](docs/Operators_ZH.md)和专用[工具池](#documentation)，旨在让数据处理独立于特定的大语言模型数据集和处理流水线。

* **数据反馈回路**：支持详细的数据分析，并提供自动报告生成功能，使您深入了解您的数据集。结合多维度自动评估功能，支持在 LLM 开发过程的多个阶段进行及时反馈循环。  ![Data-in-the-loop](https://img.alicdn.com/imgextra/i1/O1CN011E99C01ndLZ55iCUS_!!6000000005112-0-tps-2701-1050.jpg)

* **全面的数据处理菜谱**：为pre-training、fine-tuning、中英文等场景提供数十种[预构建的数据处理菜谱](configs/data_juicer_recipes/README_ZH.md)。  ![exp_llama](https://img.alicdn.com/imgextra/i2/O1CN019WtUPP1uhebnDlPR8_!!6000000006069-2-tps-2530-1005.png)

* **效率增强**：提供高效的数据处理流水线，减少内存占用和CPU开销，提高生产力。  ![sys-perf](https://img.alicdn.com/imgextra/i4/O1CN01Sk0q2U1hdRxbnQXFg_!!6000000004300-0-tps-2438-709.jpg)

* **用户友好**：设计简单易用，提供全面的[文档](#documentation)、简易[入门指南](#快速上手)和[演示配置](configs/README_ZH.md)，并且可以轻松地添加/删除[现有配置](configs/config_all.yaml)中的算子。

* **灵活 & 易扩展**：支持大多数数据格式（如jsonl、parquet、csv等），并允许灵活组合算子。支持[自定义算子](docs/DeveloperGuide_ZH.md#构建自己的算子)，以执行定制化的数据处理。


## 前置条件

* 推荐 Python==3.8
* gcc >= 5 (at least C++14 support)

## 安装

### 从源码安装

* 运行以下命令以安装 `data_juicer` 可编辑模式的最新基础版本

```shell
cd <path_to_data_juicer>
pip install -v -e .
```

* 部分算子功能依赖于较大的或者平台兼容性不是很好的第三方库，因此用户可按需额外安装可选的依赖项:

```shell
cd <path_to_data_juicer>
pip install -v -e .  # 安装最小依赖，支持基础功能
pip install -v -e .[tools] # 安装部分工具库的依赖
```

依赖选项如下表所示:

| 标签           | 描述                           |
|--------------|------------------------------|
| `.` 或者 `.[mini]` | 安装支持 Data-Juicer 基础功能的最小依赖项  |
| `.[all]`       | 安装所有可选依赖项（包括最小依赖项以及下面所有依赖项）  |
| `.[sci]`       | 安装所有算子的全量依赖                  |
| `.[dist]`      | 安装以分布式方式进行数据处理的依赖（实验性功能）     |
| `.[dev]`       | 安装作为贡献者开发 Data-Juicer 所需的依赖项 |
| `.[tools]`     | 安装专用工具库（如质量分类器）所需的依赖项        |

### 使用 pip 安装

* 运行以下命令用 `pip` 安装 `data_juicer` 的最新发布版本：

```shell
pip install py-data-juicer
```

* **注意**：
  * 使用这种方法安装时，只有`data_juicer`中的基础的 API 和2个基础工具
    （数据[处理](#数据处理)与[分析](#数据分析)）可以使用。如需更定制化地使用完整功能，建议[从源码进行安装](#从源码安装)。
  * pypi 的发布版本较源码的最新版本有一定的滞后性，如需要随时跟进 `data_juicer` 的最新功能支持，建议[从源码进行安装](#从源码安装)。

### 使用 Docker 安装

- 您可以选择
  - 从DockerHub直接拉取我们的预置镜像:
    ```shell
    docker pull datajuicer/data-juicer:<version_tag>
    ```
  - 或者运行如下命令用我们提供的 [Dockerfile](Dockerfile) 来构建包括最新版本的 `data-juicer` 的 docker 镜像：

    ```shell
    docker build -t data-juicer:<version_tag> .
    ```

### 安装校验

```python
import data_juicer as dj
print(dj.__version__)
```

## 快速上手

### 数据处理

* 以配置文件路径作为参数来运行 `process_data.py` 或者 `dj-process` 命令行工具来处理数据集。

```shell
# 适用于从源码安装
python tools/process_data.py --config configs/demo/process.yaml

# 使用命令行工具
dj-process --config configs/demo/process.yaml
```

* **注意**：使用未保存在本地的第三方模型或资源的算子第一次运行可能会很慢，因为这些算子需要将相应的资源下载到缓存目录中。默认的下载缓存目录为`~/.cache/data_juicer`。您可通过设置 shell 环境变量 `DATA_JUICER_CACHE_HOME` 更改缓存目录位置，您也可以通过同样的方式更改 `DATA_JUICER_MODELS_CACHE` 或 `DATA_JUICER_ASSETS_CACHE` 来分别修改模型缓存或资源缓存目录:

```shell
# 缓存主目录
export DATA_JUICER_CACHE_HOME="/path/to/another/directory"
# 模型缓存目录
export DATA_JUICER_MODELS_CACHE="/path/to/another/directory/models"
# 资源缓存目录
export DATA_JUICER_ASSETS_CACHE="/path/to/another/directory/assets"
```

### 数据分析

- 以配置文件路径为参数运行 `analyze_data.py` 或者 `dj-analyze` 命令行工具来分析数据集。

```shell
# 适用于从源码安装
python tools/analyze_data.py --config configs/demo/analyser.yaml

# 使用命令行工具
dj-analyze --config configs/demo/analyser.yaml
```

* **注意**：Analyser 只计算 Filter 算子的状态，其他的算子（例如 Mapper 和 Deduplicator）会在分析过程中被忽略。

### 数据可视化

* 运行 `app.py` 来在浏览器中可视化您的数据集。
* **注意**：只可用于从源码安装的方法。

```shell
streamlit run app.py
```

### 构建配置文件

* 配置文件包含一系列全局参数和用于数据处理的算子列表。您需要设置:
  * 全局参数：输入/输出 数据集路径，worker 进程数量等。
  * 算子列表：列出用于处理数据集的算子及其参数。
* 您可以通过如下方式构建自己的配置文件:
  * ➖：修改我们的样例配置文件 [`config_all.yaml`](configs/config_all.yaml)。该文件包含了**所有**算子以及算子对应的默认参数。您只需要**移除**不需要的算子并重新设置部分算子的参数即可。
  * ➕：从头开始构建自己的配置文件。您可以参考我们提供的样例配置文件 [`config_all.yaml`](configs/config_all.yaml)，[算子文档](docs/Operators_ZH.md)，以及 [开发者指南](docs/DeveloperGuide_ZH.md#构建自己的算子).
  * 除了使用 yaml 文件外，您还可以在命令行上指定一个或多个参数，这些参数将覆盖 yaml 文件中的值。

```shell
python xxx.py --config configs/demo/process.yaml --language_id_score_filter.lang=en
```

* 基础的配置项格式及定义如下图所示

  ![基础配置项格式及定义样例](https://img.alicdn.com/imgextra/i4/O1CN01xPtU0t1YOwsZyuqCx_!!6000000003050-0-tps-1692-879.jpg "基础配置文件样例")

### 预处理原始数据（可选）

* 我们的 Formatter 目前支持一些常见的输入数据集格式：
  * 单个文件中包含多个样本：jsonl/json、parquet、csv/tsv 等。
  * 单个文件中包含单个样本：txt、code、docx、pdf 等。
* 但来自不同源的数据是复杂和多样化的，例如:
  * [从 S3 下载的 arXiv 原始数据](https://info.arxiv.org/help/bulk_data_s3.html) 包括数千个 tar 文件以及更多的 gzip 文件，并且所需的 tex 文件在 gzip 文件中，很难直接获取。
  * 一些爬取的数据包含不同类型的文件（pdf、html、docx 等），并且很难提取额外的信息，例如表格、图表等。
* Data-Juicer 不可能处理所有类型的数据，欢迎提 Issues/PRs，贡献对新数据类型的处理能力！
* 因此我们在 [`tools/preprocess`](tools/preprocess) 中提供了一些**常见的预处理工具**，用于预处理这些类型各异的数据。
  * 欢迎您为社区贡献新的预处理工具。
  * 我们**强烈建议**将复杂的数据预处理为 jsonl 或 parquet 文件。

### 对于 Docker 用户

- 如果您构建或者拉取了 `data-juicer` 的 docker 镜像，您可以使用这个 docker 镜像来运行上面提到的这些命令或者工具。
- 直接运行：

```shell
# 直接运行数据处理
docker run --rm \  # 在处理结束后将容器移除
  --name dj \  # 容器名称
  -v <host_data_path>:<image_data_path> \  # 将本地的数据或者配置目录挂载到容器中
  -v ~/.cache/:/root/.cache/ \  # 将 cache 目录挂载到容器以复用 cache 和模型资源（推荐）
  data-juicer:<version_tag> \  # 运行的镜像
  dj-process --config /path/to/config.yaml  # 类似的数据处理命令
```

- 或者您可以进入正在运行的容器，然后在可编辑模式下运行命令：

```shell
# 启动容器
docker run -dit \  # 在后台启动容器
  --rm \
  --name dj \
  -v <host_data_path>:<image_data_path> \
  -v ~/.cache/:/root/.cache/ \
  data-juicer:latest /bin/bash

# 进入这个容器，然后您可以在编辑模式下使用 data-juicer
docker exec -it <container_id> bash
```

## Documentation | 文档 <a name="documentation"/>

* [Overview](README.md) | [概览](README_ZH.md)
* [Operator Zoo](docs/Operators.md) | [算子库](docs/Operators_ZH.md)
* [Configs](configs/README.md) | [配置系统](configs/README_ZH.md)
* [Developer Guide](docs/DeveloperGuide.md) | [开发者指南](docs/DeveloperGuide_ZH.md)
* Dedicated Toolkits | 专用工具箱
  * [Quality Classifier](tools/quality_classifier/README.md) | [质量分类器](tools/quality_classifier/README_ZH.md)
  * [Auto Evaluation](tools/evaluator/README.md) | [自动评测](tools/evaluator/README_ZH.md)
  * [Preprocess](tools/preprocess/README.md) | [前处理](tools/preprocess/README_ZH.md)
  * [Postprocess](tools/postprocess/README.md) | [后处理](tools/postprocess/README_ZH.md)
* [Third-parties (LLM Ecosystems)](thirdparty/README.md) | [第三方库（大语言模型生态）](thirdparty/README_ZH.md)
* [API references](https://alibaba.github.io/data-juicer/)

## 数据处理菜谱

* [BLOOM 数据处理菜谱](configs/reproduced_bloom/README_ZH.md)
* [RedPajama 数据处理菜谱](configs/reproduced_redpajama/README_ZH.md)
* [预训练数据增强菜谱](configs/data_juicer_recipes/README_ZH.md)
* [Fine-tuning数据增强菜谱](configs/data_juicer_recipes/README_ZH.md#完善前后的alpaca-cot数据集)

## 演示样例

* Data-Juicer 介绍 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/overview_scan/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/overview_scan)]
* 数据可视化:
  * 基础指标统计 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_statistics)]
  * 词汇多样性 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_diversity)]
  * 算子效果 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_effect)]
* 数据处理:
  * 科学文献 (例如 [arXiv](https://info.arxiv.org/help/bulk_data_s3.html)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_sci_data)]
  * 编程代码 (例如 [TheStack](https://huggingface.co/datasets/bigcode/the-stack)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_code_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_code_data)]
  * 中文指令数据 (例如 [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_cft_zh_data)]
* 工具池:
  * 按语言分割数据集 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language)]
  * CommonCrawl 质量分类器 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_quality_classifier)]
  * 基于 [HELM](https://github.com/stanford-crfm/helm) 的自动评测 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/auto_evaluation_helm)]
  * 数据采样及混合 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_mixture/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_mixture)]
* 数据处理回路 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_loop)]
* 数据处理 HPO [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_hpo/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_hpo)]

## 开源协议

Data-Juicer 在 Apache License 2.0 协议下发布。

## 贡献

大模型是一个高速发展的领域，我们非常欢迎贡献新功能、修复漏洞以及文档改善。请参考[开发者指南](docs/DeveloperGuide_ZH.md)。

欢迎加入我们的[Slack channel](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg?spm=a2c22.12281976.0.0.7a8275bc8g7ypp), 或[DingDing群](https://qr.dingtalk.com/action/joingroup?spm=a2c22.12281976.0.0.7a8275bc8g7ypp&code=v1,k1,C0DI7CwRFrg7gJP5aMC95FUmsNuwuKJboT62BqP5DAk=&_dt_no_comment=1&origin=11) 。

## 致谢

Data-Juicer 被各种 LLM产品和研究工作使用，包括来自阿里云-通义的行业大模型，例如点金
（金融分析），智文（阅读助手），还有阿里云人工智能平台 (PAI)。 我们期待更多您的体验反馈、建议和合作共建！


Data-Juicer 感谢并参考了社区开源项目：
[Huggingface-Datasets](https://github.com/huggingface/datasets), [Bloom](https://huggingface.co/bigscience/bloom), [RedPajama](https://github.com/togethercomputer/RedPajama-Data), [Pile](https://huggingface.co/datasets/EleutherAI/pile), [Alpaca-Cot](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [DeepSpeed](https://www.deepspeed.ai/), [Arrow](https://github.com/apache/arrow), [Ray](https://github.com/ray-project/ray), [Beam](https://github.com/apache/beam),  [LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness), [HELM](https://github.com/stanford-crfm/helm), ....



## 参考文献
如果您发现我们的工作对您的研发有帮助，请引用以下[论文](https://arxiv.org/abs/2309.02033) 。

```
@misc{chen2023datajuicer,
title={Data-Juicer: A One-Stop Data Processing System for Large Language Models},
author={Daoyuan Chen and Yilun Huang and Zhijian Ma and Hesen Chen and Xuchen Pan and Ce Ge and Dawei Gao and Yuexiang Xie and Zhaoyang Liu and Jinyang Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
year={2023},
eprint={2309.02033},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```
