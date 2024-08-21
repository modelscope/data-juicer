[[English Page]](README.md) | [[文档索引]](#documents) | [[API]](https://modelscope.github.io/data-juicer) | [[DJ-SORA]](docs/DJ_SORA_ZH.md) | [[Awesome List]](docs/awesome_llm_data.md)

# Data-Juicer: 为大模型提供更高质量、更丰富、更易“消化”的数据

 <img src="https://img.alicdn.com/imgextra/i3/O1CN017Eq5kf27AlA2NUKef_!!6000000007757-0-tps-1280-720.jpg" width = "640" height = "360" alt="Data-Juicer"/>

![](https://img.shields.io/badge/language-Python-214870.svg)
![](https://img.shields.io/badge/license-Apache--2.0-000000.svg)
[![pypi version](https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad)](https://pypi.org/project/py-data-juicer)
[![Docker version](https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf)](https://hub.docker.com/r/datajuicer/data-juicer)

[![DataModality](https://img.shields.io/badge/DataModality-Text,Image,Audio,Video-brightgreen.svg)](docs/DeveloperGuide_ZH.md)
[![Usage](https://img.shields.io/badge/Usage-Cleaning,Generation,Analysis-FFD21E.svg)](docs/DeveloperGuide_ZH.md)
[![ModelScope- Demos](https://img.shields.io/badge/ModelScope-Demos-4e29ff.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjI0IDEyMS4zMyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxwYXRoIGQ9Im0wIDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtOTkuMTQgNzMuNDloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xNzYuMDkgOTkuMTRoLTI1LjY1djIyLjE5aDQ3Ljg0di00Ny44NGgtMjIuMTl6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTEyNC43OSA0Ny44NGgyNS42NXYyNS42NWgtMjUuNjV6IiBmaWxsPSIjMzZjZmQxIiAvPgoJPHBhdGggZD0ibTAgMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xOTguMjggNDcuODRoMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xOTguMjggMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xNTAuNDQgMHYyMi4xOWgyNS42NXYyNS42NWgyMi4xOXYtNDcuODR6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTczLjQ5IDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiMzNmNmZDEiIC8+Cgk8cGF0aCBkPSJtNDcuODQgMjIuMTloMjUuNjV2LTIyLjE5aC00Ny44NHY0Ny44NGgyMi4xOXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtNDcuODQgNzMuNDloLTIyLjE5djQ3Ljg0aDQ3Ljg0di0yMi4xOWgtMjUuNjV6IiBmaWxsPSIjNjI0YWZmIiAvPgo8L3N2Zz4K)](https://modelscope.cn/studios?name=Data-Jiucer&page=1&sort=latest&type=1)
[![HuggingFace- Demos](https://img.shields.io/badge/🤗HuggingFace-Demos-4e29ff.svg)](https://huggingface.co/spaces?&search=datajuicer)

[![Document_List](https://img.shields.io/badge/Docs-English-blue?logo=Markdown)](README.md#documents)
[![文档列表](https://img.shields.io/badge/文档-中文-blue?logo=Markdown)](#documents)
[![API Reference](https://img.shields.io/badge/Docs-API_Reference-blue?logo=Markdown)](https://modelscope.github.io/data-juicer/)
[![Paper](http://img.shields.io/badge/cs.LG-arXiv%3A2309.02033-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2309.02033)


Data-Juicer 是一个一站式**多模态**数据处理系统，旨在为大语言模型 (LLM) 提供更高质量、更丰富、更易“消化”的数据。


我们提供了一个基于 JupyterLab 的 [Playground](http://8.138.149.181/)，您可以从浏览器中在线试用 Data-Juicer。 如果Data-Juicer对您的研发有帮助，请引用我们的[工作](#参考文献) 。

Data-Juicer正在积极更新和维护中，我们将定期强化和新增更多的功能和数据菜谱。热烈欢迎您加入我们（issues/PRs/[Slack频道](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg?spm=a2c22.12281976.0.0.7a8275bc8g7ypp) /[钉钉群](https://qr.dingtalk.com/action/joingroup?spm=a2c22.12281976.0.0.7a8275bc8g7ypp&code=v1,k1,C0DI7CwRFrg7gJP5aMC95FUmsNuwuKJboT62BqP5DAk=&_dt_no_comment=1&origin=11)/...），一起推进LLM-数据的协同开发和研究！


----

## 新消息
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-08-09] 我们提出了Img-Diff，它通过*对比数据合成*来增强多模态大型语言模型的性能，在[MMVP benchmark](https://tsb0601.github.io/mmvp_blog/)中比GPT-4V高出12个点。 更多细节请参阅我们的 [论文](https://arxiv.org/abs/2408.04594), 以及从 [huggingface](https://huggingface.co/datasets/datajuicer/Img-Diff) 和 [modelscope](https://modelscope.cn/datasets/Data-Juicer/Img-Diff)下载这份数据集。
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-07-24] “天池 Better Synth 多模态大模型数据合成赛”——第四届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532251)，了解赛事详情。
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)[2024-07-17] 我们利用Data-Juicer[沙盒实验室套件](https://github.com/modelscope/data-juicer/blob/main/docs/Sandbox-ZH.md)，通过数据与模型间的系统性研发工作流，调优数据和模型，在[VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)文生视频排行榜取得了新的榜首。相关成果已经整理发表在[论文](http://arxiv.org/abs/2407.11784)中，并且模型已在[ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V)和[HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V)平台发布。
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)[2024-07-12] 我们的MLLM-Data精选列表已经演化为一个模型-数据协同开发的角度系统性[综述](https://arxiv.org/abs/2407.08583)。欢迎[浏览](docs/awesome_llm_data.md)或参与贡献!
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-06-01] ModelScope-Sora“数据导演”创意竞速——第三届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532219)，了解赛事详情。
- [2024-03-07] 我们现在发布了 **Data-Juicer [v0.2.0](https://github.com/alibaba/data-juicer/releases/tag/v0.2.0)**! 在这个新版本中，我们支持了更多的 **多模态数据(包括视频)** 相关特性。我们还启动了 **[DJ-SORA](docs/DJ_SORA_ZH.md)** ，为SORA-like大模型构建开放的大规模高质量数据集！
- [2024-02-20] 我们在积极维护一份关于LLM-Data的*精选列表*，欢迎[访问](docs/awesome_llm_data.md)并参与贡献！
- [2024-02-05] 我们的论文被SIGMOD'24 industrial track接收！
- [2024-01-10] 开启“数据混合”新视界——第二届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532174)，了解赛事详情。
- [2024-01-05] **Data-Juicer v0.1.3** 版本发布了。 
在这个新版本中，我们支持了**更多Python版本**（3.8-3.10），同时支持了**多模态**数据集的[转换](tools/multimodal/README_ZH.md)和[处理](docs/Operators_ZH.md)（包括文本、图像和音频。更多模态也将会在之后支持）！
此外，我们的论文也更新到了[第三版](https://arxiv.org/abs/2309.02033) 。
- [2023-10-13] 我们的第一届以数据为中心的 LLM 竞赛开始了！
  请访问大赛官网，FT-Data Ranker（[1B赛道](https://tianchi.aliyun.com/competition/entrance/532157) 、[7B赛道](https://tianchi.aliyun.com/competition/entrance/532158) ) ，了解更多信息。

<div id="table" align="center"></div>

目录
===
- [Data-Juicer: 为大语言模型提供更高质量、更丰富、更易“消化”的数据](#data-juicer-为大语言模型提供更高质量更丰富更易消化的数据)
  - [新消息](#新消息)
- [目录](#目录)
  - [特点](#特点)
  - [文档索引 ](#文档索引-)
  - [演示样例](#演示样例)
  - [前置条件](#前置条件)
  - [安装](#安装)
    - [从源码安装](#从源码安装)
    - [使用 pip 安装](#使用-pip-安装)
    - [使用 Docker 安装](#使用-docker-安装)
    - [安装校验](#安装校验)
  - [快速上手](#快速上手)
    - [数据处理](#数据处理)
    - [分布式数据处理](#分布式数据处理)
    - [数据分析](#数据分析)
    - [数据可视化](#数据可视化)
    - [构建配置文件](#构建配置文件)
    - [沙盒实验室](#沙盒实验室)
    - [预处理原始数据（可选）](#预处理原始数据可选)
    - [对于 Docker 用户](#对于-docker-用户)
  - [数据处理菜谱](#数据处理菜谱)
  - [开源协议](#开源协议)
  - [贡献](#贡献)
  - [致谢](#致谢)
  - [参考文献](#参考文献)


## 特点

![Overview](https://img.alicdn.com/imgextra/i4/O1CN01WYQP3Z1JHsaXaQDK6_!!6000000001004-0-tps-3640-1812.jpg)

* **系统化 & 可复用**：为用户提供系统化且可复用的80+核心[算子](docs/Operators_ZH.md)，20+[配置菜谱](configs/README_ZH.md)和20+专用[工具池](#documentation)，旨在让多模态数据处理独立于特定的大语言模型数据集和处理流水线。

* **数据反馈回路 & 沙盒实验室**：支持一站式数据-模型协同开发，通过[沙盒实验室](docs/Sandbox-ZH.md)快速迭代，基于数据和模型反馈回路、可视化和多维度自动评估等功能，使您更了解和改进您的数据和模型。  ![Data-in-the-loop](https://img.alicdn.com/imgextra/i2/O1CN017U7Zz31Y7XtCJ5GOz_!!6000000003012-0-tps-3640-1567.jpg)

* **面向生产环境**：提供高效并行化的数据处理流水线（Aliyun-PAI\Ray\Slurm\CUDA\算子融合），减少内存占用和CPU开销，支持自动化处理容错。  ![sys-perf](https://img.alicdn.com/imgextra/i4/O1CN01Sk0q2U1hdRxbnQXFg_!!6000000004300-0-tps-2438-709.jpg)

* **全面的数据处理菜谱**：为pre-training、fine-tuning、中英文等场景提供数十种[预构建的数据处理菜谱](configs/data_juicer_recipes/README_ZH.md)。 在LLaMA、LLaVA等模型上有效验证。 ![exp_llama](https://img.alicdn.com/imgextra/i2/O1CN019WtUPP1uhebnDlPR8_!!6000000006069-2-tps-2530-1005.png)

* **用户友好**：设计简单易用，提供全面的[文档](#documents)、简易[入门指南](#快速上手)和[演示配置](configs/README_ZH.md)，并且可以轻松地添加/删除[现有配置](configs/config_all.yaml)中的算子。

* **灵活 & 易扩展**：支持大多数数据格式（如jsonl、parquet、csv等），并允许灵活组合算子。支持[自定义算子](docs/DeveloperGuide_ZH.md#构建自己的算子)，以执行定制化的数据处理。


## 文档索引 <a name="documents"/>

* [概览](README_ZH.md)
* [算子库](docs/Operators_ZH.md)
* [配置系统](configs/README_ZH.md)
* [开发者指南](docs/DeveloperGuide_ZH.md)
* [“坏”数据展览](docs/BadDataExhibition_ZH.md)
* 专用工具箱
  * [质量分类器](tools/quality_classifier/README_ZH.md)
  * [自动评测](tools/evaluator/README_ZH.md)
  * [前处理](tools/preprocess/README_ZH.md)
  * [后处理](tools/postprocess/README_ZH.md)
* [第三方库（大语言模型生态）](thirdparty/README_ZH.md)
* [API 参考](https://modelscope.github.io/data-juicer/)
* [Awesome LLM-Data](docs/awesome_llm_data.md)
* [DJ-SORA](docs/DJ_SORA_ZH.md)


## 演示样例

* Data-Juicer 介绍 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/overview_scan/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/overview_scan)]
* 数据可视化:
  * 基础指标统计 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_statistics)]
  * 词汇多样性 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_diversity)]
  * 算子洞察（单OP） [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visualization_op_insight/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_insight)]
  * 算子效果（多OP） [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_effect)]
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


## 前置条件

* 推荐 Python>=3.8,<=3.10
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

| 标签               | 描述                           |
|------------------|------------------------------|
| `.` 或者 `.[mini]` | 安装支持 Data-Juicer 基础功能的最小依赖项  |
| `.[all]`         | 安装除了沙盒实验以外的所有依赖项  |
| `.[sci]`         | 安装所有算子的全量依赖                  |
| `.[dist]`        | 安装以分布式方式进行数据处理的依赖（实验性功能）     |
| `.[dev]`         | 安装作为贡献者开发 Data-Juicer 所需的依赖项 |
| `.[tools]`       | 安装专用工具库（如质量分类器）所需的依赖项        |
| `.[sandbox]`     | 安装沙盒实验室的基础依赖                 |

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
    docker build -t datajuicer/data-juicer:<version_tag> .
    ```

  - `<version_tag>`的格式类似于`v0.2.0`，与发布（Release）的版本号相同。

### 安装校验

```python
import data_juicer as dj
print(dj.__version__)
```

### 使用视频相关算子

在使用视频相关算子之前，应该安装 **FFmpeg** 并确保其可通过 $PATH 环境变量访问。

你可以使用包管理器安装 FFmpeg（例如，在 Debian/Ubuntu 上使用 sudo apt install ffmpeg，在 OS X 上使用 brew install ffmpeg），或访问[官方FFmpeg链接](https://ffmpeg.org/download.html)。

随后在终端运行 ffmpeg 命令检查环境是否设置正确。


<p align="right"><a href="#table">🔼 back to index</a></p>

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

* **注意**：对于使用了第三方模型的算子，在填写config文件时需要去声明其对应的`mem_required`（可以参考`config_all.yaml`文件中的设置）。Data-Juicer在运行过程中会根据内存情况和算子模型所需的memory大小来控制对应的进程数，以达成更好的数据处理的性能效率。而在使用CUDA环境运行时，如果不正确的声明算子的`mem_required`情况，则有可能导致CUDA Out of Memory。

```shell
# 缓存主目录
export DATA_JUICER_CACHE_HOME="/path/to/another/directory"
# 模型缓存目录
export DATA_JUICER_MODELS_CACHE="/path/to/another/directory/models"
# 资源缓存目录
export DATA_JUICER_ASSETS_CACHE="/path/to/another/directory/assets"
```

#### 灵活的编程接口
我们提供了各种层次的简单编程接口，以供用户选择：
```python
# ... init op & dataset ...

# 链式调用风格，支持单算子或算子列表
dataset = dataset.process(op)
dataset = dataset.process([op1, op2])
# 函数式编程风格，方便快速集成或脚本原型迭代
dataset = op(dataset)
dataset = op.run(dataset)
```

### 分布式数据处理

Data-Juicer 现在基于[RAY](https://www.ray.io/)实现了多机分布式数据处理。
对应Demo可以通过如下命令运行：

```shell

# 运行文字数据处理
python tools/process_data.py --config ./demos/process_on_ray/configs/demo.yaml

# 运行视频数据处理
python tools/process_data.py --config ./demos/process_video_on_ray/configs/demo.yaml

```

 - 如果需要在多机上使用RAY执行数据处理，需要确保所有节点都可以访问对应的数据路径，即将对应的数据路径挂载在共享文件系统（如NAS）中。
 - RAY 模式下的去重算子与单机版本不同，所有 RAY 模式下的去重算子名称都以 `ray` 作为前缀，例如 `ray_video_deduplicator` 和 `ray_document_deduplicator`。这些去重算子依赖于 [Redis](https://redis.io/) 实例.因此使用前除启动 RAY 集群外还需要启动 Redis 实例，并在对应的配置文件中填写 Redis 实例的 `host` 和 `port`。

> 用户也可以不使用 RAY，拆分数据集后使用 [Slurm](https://slurm.schedmd.com/) / [阿里云 PAI-DLC](https://www.aliyun.com/activity/bigdata/pai-dlc) 在集群上运行，此时使用不包含 RAY 的原版 Data-Juicer 即可。

### 数据分析

- 以配置文件路径为参数运行 `analyze_data.py` 或者 `dj-analyze` 命令行工具来分析数据集。

```shell
# 适用于从源码安装
python tools/analyze_data.py --config configs/demo/analyzer.yaml

# 使用命令行工具
dj-analyze --config configs/demo/analyzer.yaml
```

* **注意**：Analyzer 只计算 Filter 算子的状态，其他的算子（例如 Mapper 和 Deduplicator）会在分析过程中被忽略。

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

### 沙盒实验室

数据沙盒实验室 (DJ-Sandbox) 为用户提供了持续生产数据菜谱的最佳实践，其具有低开销、可迁移、有指导性等特点。
- 用户在沙盒中可以基于一些小规模数据集、模型对数据菜谱进行快速实验、迭代、优化，再迁移到更大尺度上，大规模生产高质量数据以服务大模型。
- 用户在沙盒中，除了Data-Juicer基础的数据优化与数据菜谱微调功能外，还可以便捷地使用数据洞察与分析、沙盒模型训练与评测、基于数据和模型反馈优化数据菜谱等可配置组件，共同组成完整的一站式数据-模型研发流水线。

沙盒默认通过如下命令运行，更多介绍和细节请参阅[沙盒文档](docs/Sandbox-ZH.md).
```shell
python tools/sandbox_starter.py --config configs/demo/sandbox/sandbox.yaml
```



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
  datajuicer/data-juicer:<version_tag> \  # 运行的镜像
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
  datajuicer/data-juicer:latest /bin/bash

# 进入这个容器，然后您可以在编辑模式下使用 data-juicer
docker exec -it <container_id> bash
```


<p align="right"><a href="#table">🔼 back to index</a></p>

## 数据处理菜谱

* [BLOOM 数据处理菜谱](configs/reproduced_bloom/README_ZH.md)
* [RedPajama 数据处理菜谱](configs/reproduced_redpajama/README_ZH.md)
* [预训练文本数据增强菜谱](configs/data_juicer_recipes/README_ZH.md)
* [Fine-tuning文本数据增强菜谱](configs/data_juicer_recipes/README_ZH.md#完善前后的alpaca-cot数据集)
* [预训练多模态数据增强菜谱](configs/data_juicer_recipes/README_ZH.md#before-and-after-refining-for-multimodal-dataset)

## 开源协议

Data-Juicer 在 Apache License 2.0 协议下发布。

## 贡献

大模型是一个高速发展的领域，我们非常欢迎贡献新功能、修复漏洞以及文档改善。请参考[开发者指南](docs/DeveloperGuide_ZH.md)。

如果您有任何问题，欢迎加入我们的[讨论群](README_ZH.md) 。

## 致谢

Data-Juicer 被各种 LLM产品和研究工作使用，包括来自阿里云-通义的行业大模型，例如点金
（金融分析），智文（阅读助手），还有阿里云人工智能平台 (PAI)。 我们期待更多您的体验反馈、建议和合作共建！


Data-Juicer 感谢并参考了社区开源项目：
[Huggingface-Datasets](https://github.com/huggingface/datasets), [Bloom](https://huggingface.co/bigscience/bloom), [RedPajama](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1), [Pile](https://huggingface.co/datasets/EleutherAI/pile), [Alpaca-Cot](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [DeepSpeed](https://www.deepspeed.ai/), [Arrow](https://github.com/apache/arrow), [Ray](https://github.com/ray-project/ray), [Beam](https://github.com/apache/beam),  [LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness), [HELM](https://github.com/stanford-crfm/helm), ....

## 参考文献
如果您发现我们的工作对您的研发有帮助，请引用以下[论文](https://arxiv.org/abs/2309.02033) 。

```
@inproceedings{chen2024datajuicer,
  title={Data-Juicer: A One-Stop Data Processing System for Large Language Models},
  author={Daoyuan Chen and Yilun Huang and Zhijian Ma and Hesen Chen and Xuchen Pan and Ce Ge and Dawei Gao and Yuexiang Xie and Zhaoyang Liu and Jinyang Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
  booktitle={International Conference on Management of Data},
  year={2024}
}
```
<details>
<summary>更多Data-Juicer团队相关论文:
</summary>>

- [Data-Juicer Sandbox: A Comprehensive Suite for Multimodal Data-Model Co-development](https://arxiv.org/abs/2407.11784)

- [The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective](https://arxiv.org/abs/2407.08583)

- [ImgDiff: Contrastive Data Synthesis for Vision Large Language Models](https://arxiv.org/abs/2408.04594)

- [Data Mixing Made Efficient: A Bivariate Scaling Law for Language Model Pretraining](https://arxiv.org/abs/2402.11505)

</details>



<p align="right"><a href="#table">🔼 back to index</a></p>