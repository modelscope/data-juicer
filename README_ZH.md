[[英文主页]](README.md) | [[DJ-Cookbook]](#dj-cookbook) | [[算子池]](docs/Operators.md) | [[API]](https://modelscope.github.io/data-juicer) | [[Awesome LLM Data]](docs/awesome_llm_data.md)

# Data Processing for and with Foundation Models

 <img src="https://img.alicdn.com/imgextra/i1/O1CN01fUfM5A1vPclzPQ6VI_!!6000000006165-0-tps-1792-1024.jpg" width = "533" height = "300" alt="Data-Juicer"/>

![](https://img.shields.io/badge/language-Python-214870.svg)
![](https://img.shields.io/badge/license-Apache--2.0-000000.svg)
[![pypi version](https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad)](https://pypi.org/project/py-data-juicer)
[![Docker version](https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf)](https://hub.docker.com/r/datajuicer/data-juicer)
[![Docker on OSS](https://img.shields.io/badge/OSS%20latest-none?logo=docker&label=Docker&color=498bdf)](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/docker_images/data-juicer-latest.tar.gz)
![](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FHYLcool%2Ff856b14416f08f73d05d32fd992a9c29%2Fraw%2Ftotal_cov.json)

[![DataModality](https://img.shields.io/badge/DataModality-Text,Image,Audio,Video-brightgreen.svg)](#dj-cookbook)
[![Usage](https://img.shields.io/badge/Usage-Cleaning,Synthesis,Analysis-FFD21E.svg)](#dj-cookbook)
[![ModelScope- Demos](https://img.shields.io/badge/ModelScope-Demos-4e29ff.svg?logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjI0IDEyMS4zMyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxwYXRoIGQ9Im0wIDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtOTkuMTQgNzMuNDloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xNzYuMDkgOTkuMTRoLTI1LjY1djIyLjE5aDQ3Ljg0di00Ny44NGgtMjIuMTl6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTEyNC43OSA0Ny44NGgyNS42NXYyNS42NWgtMjUuNjV6IiBmaWxsPSIjMzZjZmQxIiAvPgoJPHBhdGggZD0ibTAgMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xOTguMjggNDcuODRoMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzYyNGFmZiIgLz4KCTxwYXRoIGQ9Im0xOTguMjggMjIuMTloMjUuNjV2MjUuNjVoLTI1LjY1eiIgZmlsbD0iIzM2Y2ZkMSIgLz4KCTxwYXRoIGQ9Im0xNTAuNDQgMHYyMi4xOWgyNS42NXYyNS42NWgyMi4xOXYtNDcuODR6IiBmaWxsPSIjNjI0YWZmIiAvPgoJPHBhdGggZD0ibTczLjQ5IDQ3Ljg0aDI1LjY1djI1LjY1aC0yNS42NXoiIGZpbGw9IiMzNmNmZDEiIC8+Cgk8cGF0aCBkPSJtNDcuODQgMjIuMTloMjUuNjV2LTIyLjE5aC00Ny44NHY0Ny44NGgyMi4xOXoiIGZpbGw9IiM2MjRhZmYiIC8+Cgk8cGF0aCBkPSJtNDcuODQgNzMuNDloLTIyLjE5djQ3Ljg0aDQ3Ljg0di0yMi4xOWgtMjUuNjV6IiBmaWxsPSIjNjI0YWZmIiAvPgo8L3N2Zz4K)](https://modelscope.cn/studios?name=Data-Jiucer&page=1&sort=latest&type=1)
[![HuggingFace- Demos](https://img.shields.io/badge/🤗HuggingFace-Demos-4e29ff.svg)](https://huggingface.co/spaces?&search=datajuicer)

[![Document_List](https://img.shields.io/badge/Doc-DJ_Cookbook-blue?logo=Markdown)](#dj-cookbook)
[![文档列表](https://img.shields.io/badge/文档-DJ指南-blue?logo=Markdown)](README_ZH.md#dj-cookbook)
[![算子池](https://img.shields.io/badge/文档-算子池-blue?logo=Markdown)](docs/Operators.md)
[![Paper](http://img.shields.io/badge/cs.LG-1.0Paper(SIGMOD'24)-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2309.02033)
[![Paper](http://img.shields.io/badge/cs.AI-2.0Paper-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2501.14755)



Data-Juicer 是一个一站式系统，面向大模型的文本及多模态数据处理。我们提供了一个基于 JupyterLab 的 [Playground](http://8.138.149.181/)，您可以从浏览器中在线试用 Data-Juicer。 如果Data-Juicer对您的研发有帮助，请支持加星（自动订阅我们的新发布）、以及引用我们的[工作](#参考文献) 。

[阿里云人工智能平台 PAI](https://www.aliyun.com/product/bigdata/learn) 已引用Data-Juicer并将其能力集成到PAI的数据处理产品中。PAI提供包含数据集管理、算力管理、模型工具链、模型开发、模型训练、模型部署、AI资产管理在内的功能模块，为用户提供高性能、高稳定、企业级的大模型工程化能力。数据处理的使用文档请参考：[PAI-大模型数据处理](https://help.aliyun.com/zh/pai/user-guide/components-related-to-data-processing-for-foundation-models/?spm=a2c4g.11186623.0.0.3e9821a69kWdvX)。

Data-Juicer正在积极更新和维护中，我们将定期强化和新增更多的功能和数据菜谱。热烈欢迎您加入我们（issues/PRs/[Slack频道](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg?spm=a2c22.12281976.0.0.7a8275bc8g7ypp) /[钉钉群](https://qr.dingtalk.com/action/joingroup?code=v1,k1,YFIXM2leDEk7gJP5aMC95AfYT+Oo/EP/ihnaIEhMyJM=&_dt_no_comment=1&origin=11)/...），一起推进大模型的数据-模型协同开发和研究应用！


----

## 新消息
- 🎉 [2025-05-06] 我们的 [Data-Juicer Sandbox](https://arxiv.org/abs/2407.11784) 已被 *ICML 2025* 接收为 **Spotlight**（处于所有投稿中的前 2.6%）！
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2025-03-13] 我们提出了一种新的数据合成方法 *MindGym*，该方法鼓励 LLM 自我生成具有挑战性的认知问题，实现优于 SOTA 基线的数据效率、跨模态泛化和 SFT 效果（例如，仅使用 *400 个样本* 即可在 [MathVision](https://mathllm.github.io/mathvision/#leaderboard) 上获得 *16%* 的增益）。有关更多详细信息，请参阅[MindGym: Enhancing Vision-Language Models via Synthetic Self-Challenging Questions](https://arxiv.org/abs/2503.09499)。
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2025-02-28] DJ 已被集成到 [Ray官方 Ecosystem](https://docs.ray.io/en/latest/ray-overview/ray-libraries.html) 和 [Example Gallery](https://docs.ray.io/en/latest/ray-more-libs/data_juicer_distributed_data_processing.html)。此外，我们在 DJ2.0 中的流式 JSON 加载补丁已被 [Apache Arrow 官方集成](https://github.com/apache/arrow/pull/45084)。
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2025-02-27] 我们的对比数据合成工作， [ImgDiff](https://arxiv.org/pdf/2408.04594)， 已被 *CVPR 2025* 接收！
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2025-02-05] 我们提出了一种新的数据选择方法 *DaaR*，该方法基于理论指导，将数据多样性建模为奖励信号，在 7 个基准测试中，微调 SOTA LLMs 取得了更好的整体表现。有关更多详细信息，请参阅 [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://www.arxiv.org/abs/2502.04380) 。
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2025-01-11] 我们发布了 2.0 版论文 [Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for Foundation Models](https://arxiv.org/abs/2501.14755)。DJ现在可以使用阿里云集群中 50 个 Ray 节点上的 6400 个 CPU 核心在 2.1 小时内处理 70B 数据样本，并使用 8 个 Ray 节点上的 1280 个 CPU 核心在 2.8 小时内对 5TB 数据进行重复数据删除。
- ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2025-01-03] 我们通过 20 多个相关的新 [OP](https://github.com/modelscope/data-juicer/releases/tag/v1.0.2) 以及与 LLaMA-Factory 和 ModelScope-Swift 兼容的统一 [数据集格式](https://github.com/modelscope/data-juicer/releases/tag/v1.0.3) 更好地支持Post-Tuning场景。

<details>
<summary> History News:
</summary>>

- [2024-12-17] 我们提出了 *HumanVBench*，它包含 16 个以人为中心的任务，使用合成数据，从内在情感和外在表现的角度对22个视频 MLLM 的能力进行基准测试。请参阅我们的 [论文](https://arxiv.org/abs/2412.17574) 中的更多详细信息，并尝试使用它 [评估](https://github.com/modelscope/data-juicer/tree/HumanVBench) 您的模型。

- [2024-11-22] 我们发布 DJ [v1.0.0](https://github.com/modelscope/data-juicer/releases/tag/v1.0.0)，其中我们重构了 Data-Juicer 的 *Operator*、*Dataset*、*Sandbox* 和许多其他模块以提高可用性，例如支持容错、FastAPI 和自适应资源管理。

- [2024-08-25] 我们在 KDD'2024 中提供了有关多模态 LLM 数据处理的[教程](https://modelscope.github.io/data-juicer/_static/tutorial_kdd24.html)。

- [2024-08-09] 我们提出了Img-Diff，它通过*对比数据合成*来增强多模态大型语言模型的性能，在[MMVP benchmark](https://tsb0601.github.io/mmvp_blog/)中比GPT-4V高出12个点。 更多细节请参阅我们的 [论文](https://arxiv.org/abs/2408.04594), 以及从 [huggingface](https://huggingface.co/datasets/datajuicer/Img-Diff) 和 [modelscope](https://modelscope.cn/datasets/Data-Juicer/Img-Diff)下载这份数据集。
- [2024-07-24] “天池 Better Synth 多模态大模型数据合成赛”——第四届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532251)，了解赛事详情。
- [2024-07-17] 我们利用Data-Juicer[沙盒实验室套件](https://github.com/modelscope/data-juicer/blob/main/docs/Sandbox_ZH.md)，通过数据与模型间的系统性研发工作流，调优数据和模型，在[VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)文生视频排行榜取得了新的榜首。相关成果已经整理发表在[论文](http://arxiv.org/abs/2407.11784)中，并且模型已在[ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V)和[HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V)平台发布。
- [2024-07-12] 我们的MLLM-Data精选列表已经演化为一个模型-数据协同开发的角度系统性[综述](https://arxiv.org/abs/2407.08583)。欢迎[浏览](docs/awesome_llm_data.md)或参与贡献!
- [2024-06-01] ModelScope-Sora“数据导演”创意竞速——第三届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532219)，了解赛事详情。
- [2024-03-07] 我们现在发布了 **Data-Juicer [v0.2.0](https://github.com/alibaba/data-juicer/releases/tag/v0.2.0)**! 在这个新版本中，我们支持了更多的 **多模态数据(包括视频)** 相关特性。我们还启动了 **[DJ-SORA](docs/DJ_SORA_ZH.md)** ，为SORA-like大模型构建开放的大规模高质量数据集！
- [2024-02-20] 我们在积极维护一份关于LLM-Data的*精选列表*，欢迎[访问](docs/awesome_llm_data.md)并参与贡献！
- [2024-02-05] 我们的论文被SIGMOD'24 industrial track接收！
- [2024-01-10] 开启“数据混合”新视界——第二届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532174)，了解赛事详情。
- [2024-01-05] **Data-Juicer v0.1.3** 版本发布了。 
在这个新版本中，我们支持了**更多Python版本**（3.8-3.10），同时支持了**多模态**数据集的[转换](tools/fmt_conversion/multimodal/README_ZH.md)和[处理](docs/Operators.md)（包括文本、图像和音频。更多模态也将会在之后支持）！
此外，我们的论文也更新到了[第三版](https://arxiv.org/abs/2309.02033) 。
- [2023-10-13] 我们的第一届以数据为中心的 LLM 竞赛开始了！
  请访问大赛官网，FT-Data Ranker（[1B赛道](https://tianchi.aliyun.com/competition/entrance/532157) 、[7B赛道](https://tianchi.aliyun.com/competition/entrance/532158) ) ，了解更多信息。
</details>


<div id="table" align="center"></div>

目录
===
- [新消息](#新消息)
- [为什么选择 Data-Juicer？](#为什么选择-data-juicer)
- [DJ-Cookbook](#dj-cookbook)
  - [资源合集](#资源合集)
  - [编写Data-Juicer (DJ) 代码](#编写data-juicer-dj-代码)
  - [用例与数据菜谱](#用例与数据菜谱)
  - [交互类示例](#交互类示例)
- [安装](#安装)
  - [前置条件](#前置条件)
  - [从源码安装 (指定使用场景)](#从源码安装-指定使用场景)
  - [从源码安装 (指定部分算子)](#从源码安装-指定部分算子)
  - [使用 pip 安装](#使用-pip-安装)
  - [使用 Docker 安装](#使用-docker-安装)
  - [安装校验](#安装校验)
  - [使用视频相关算子](#使用视频相关算子)
- [快速上手](#快速上手)
  - [数据集配置](#数据集配置)
  - [数据处理](#数据处理)
  - [分布式数据处理](#分布式数据处理)
  - [数据分析](#数据分析)
  - [数据可视化](#数据可视化)
  - [构建配置文件](#构建配置文件)
  - [沙盒实验室](#沙盒实验室)
  - [预处理原始数据（可选）](#预处理原始数据可选)
  - [对于 Docker 用户](#对于-docker-用户)
- [开源协议](#开源协议)
- [贡献](#贡献)
- [致谢](#致谢)
- [参考文献](#参考文献)


## 为什么选择 Data-Juicer？

<img src="https://img.alicdn.com/imgextra/i2/O1CN01EteoQ31taUweAW1UE_!!6000000005918-2-tps-4034-4146.png" align="center" width="600" />

- **系统化和可重用**：
系统化地为用户提供 100 多个核心 [算子](docs/Operators.md) 和 50 多个可重用的数据菜谱和
专用工具套件，旨在解耦于特定的多模态 LLM 数据集和处理管道运行。支持预训练、后训练、英语、中文等场景中的数据分析、清洗和合成。

- **易用、可扩展**：
简洁灵活，提供快速[入门指南](#快速上手)和包含丰富使用示例的[DJ-Cookbook](#dj-cookbook)。您可以灵活实现自己的OP，[自定义](docs/DeveloperGuide_ZH.md)数据处理工作流。

- **高效、稳定**：提供性能优化的[并行数据处理能力](docs/Distributed_ZH.md)（Aliyun-PAI\Ray\CUDA\OP Fusion），
更快、更少资源消耗，基于大规模生产环境打磨。

- **效果验证、沙盒**：支持数据模型协同开发，通过[沙盒实验室](docs/Sandbox_ZH.md)实现快速迭代，提供反馈循环、可视化等功能，让您更好地理解和改进数据和模型。已经有许多基于 DJ 衍生的数据菜谱和模型经过了效用验证，譬如在预训练、文生视频、图文生成等场景。
![Data-in-the-loop](https://img.alicdn.com/imgextra/i2/O1CN017U7Zz31Y7XtCJ5GOz_!!6000000003012-0-tps-3640-1567.jpg)

## DJ-Cookbook
### 资源合集
- [KDD'24 相关教程](https://modelscope.github.io/data-juicer/_static/tutorial_kdd24.html)
- [Awesome LLM-Data](docs/awesome_llm_data.md)
- [“坏”数据展览](docs/BadDataExhibition_ZH.md)

### 编写Data-Juicer (DJ) 代码
- 基础
  - [DJ概览](README_ZH.md)
  - [快速上手](#快速上手)
  - [配置](docs/RecipeGallery_ZH.md)
  - [数据格式转换](tools/fmt_conversion/README_ZH.md)
- 信息速查
  - [算子库](docs/Operators.md)
  - [API参考](https://modelscope.github.io/data-juicer/)
- 进阶
  - [开发者指南](docs/DeveloperGuide_ZH.md)
  - [预处理工具](tools/preprocess/README_ZH.md)
  - [后处理工具](tools/postprocess/README_ZH.md)
  - [沙盒](docs/Sandbox_ZH.md)
  - [API服务化](docs/DJ_service_ZH.md)
  - [给数据打分](tools/quality_classifier/README_ZH.md)
  - [自动评估](tools/evaluator/README_ZH.md)
  - [第三方集成](thirdparty/LLM_ecosystems/README_ZH.md)

### 用例与数据菜谱
* [数据菜谱Gallery](docs/RecipeGallery.md)
  - Data-Juicer 最小示例配方
  - 复现开源文本数据集
  - 改进开源文本预训练数据集
  - 改进开源文本后处理数据集
  - 合成对比学习图像文本数据集
  - 改进开源图像文本数据集
  - 视频数据的基本示例菜谱
  - 合成以人为中心的视频评测集
  - 改进现有的开源视频数据集
* Data-Juicer相关竞赛
  - [Better Synth](https://tianchi.aliyun.com/competition/entrance/532251)，在DJ-沙盒实验室和多模态大模型上，探索大模型合成数据对图像理解能力的影响
  - [Modelscope-Sora挑战赛](https://tianchi.aliyun.com/competition/entrance/532219)，基于Data-Juicer和[EasyAnimate](https://github.com/aigc-apps/EasyAnimate)框架，调优文本-视频数据集，在类SORA小模型上训练以生成更好的视频
  - [Better Mixture](https://tianchi.aliyun.com/competition/entrance/532174)，针对指定多个候选数据集，仅调整数据混合和采样策略
  - FT-Data Ranker ([1B Track](https://tianchi.aliyun.com/competition/entrance/532157)、 [7B Track](https://tianchi.aliyun.com/competition/entrance/532158))，针对指定候选数据集，仅调整数据过滤和增强策略
  - [可图Kolors-LoRA风格故事挑战赛](https://tianchi.aliyun.com/competition/entrance/532254)，基于Data-Juicer和[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)框架，探索Diffusion模型微调
* [DJ-SORA](docs/DJ_SORA_ZH.md)
* 基于Data-Juicer和[AgentScope](https://github.com/modelscope/agentscope)框架，通过[智能体调用DJ Filters](./demos/api_service/react_data_filter_process.ipynb)和[调用DJ Mappers](./demos/api_service/react_data_mapper_process.ipynb)
  


### 交互类示例
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


## 安装

### 前置条件

* 推荐 Python>=3.9,<=3.10
* gcc >= 5 (at least C++14 support)


### 从源码安装 (指定使用场景)

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

| 标签              | 描述 |
|------------------|----------------------------------|
| `.` 或 `.[mini]` | 为基本 Data-Juicer 安装最小依赖项。  |
| `.[all]`         | 为除沙盒之外的所有 OP 安装依赖项。    |
| `.[sci]`         | 为与科学用途相关的 OP 安装依赖项。    |
| `.[dist]`        | 安装用于分布式数据处理的额外依赖项。   |
| `.[dev]`         | 安装作为贡献者开发软件包的依赖项。     |
| `.[tools]`       | 安装专用工具（例如质量分类器）的依赖项。|
| `.[sandbox]`     | 安装沙盒的所有依赖项。               |

### 从源码安装 (指定部分算子)

* 只安装部分算子依赖

随着OP数量的增长，全OP环境的依赖安装会变得越来越重。为此，我们提供了两个替代的、更轻量的选项，作为使用命令`pip install -v -e .[sci]`安装所有依赖的替代：

  * 自动最小依赖安装：在执行Data-Juicer的过程中，将自动安装最小依赖。也就是说你可以安装mini后直接执行，但这种方式可能会导致一些(滞后的)依赖冲突。

  * 手动最小依赖安装：可以通过如下指令手动安装适合特定执行配置的最小依赖，可以提前确定依赖冲突、使其更易解决:
    ```shell
    # 从源码安装
    python tools/dj_install.py --config path_to_your_data-juicer_config_file
    
    # 使用命令行工具
    dj-install --config path_to_your_data-juicer_config_file
    ```

### 使用 pip 安装

* 运行以下命令用 `pip` 安装 `data_juicer` 的最新发布版本：

```shell
pip install py-data-juicer
```

* **注意**：
  * 使用这种方法安装时，只有`data_juicer`中的基础的 API 和2个基础工具
    （数据[处理](#数据处理)与[分析](#数据分析)）可以使用。如需更定制化地使用完整功能，建议[从源码进行安装](#从源码安装-指定使用场景)。
  * pypi 的发布版本较源码的最新版本有一定的滞后性，如需要随时跟进 `data_juicer` 的最新功能支持，建议[从源码进行安装](#从源码安装-指定使用场景)。

### 使用 Docker 安装

- 您可以选择
  - 从DockerHub直接拉取我们的预置镜像:
    ```shell
    docker pull datajuicer/data-juicer:<version_tag>
    ```
    
    - 如您无法连接到DockerHub，请使用其他可用的Docker镜像源拉取（可从互联网搜索获取）：
    ```shell
    docker pull <其他可用镜像源>/datajuicer/data-juicer:<version_tag>
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
### 数据集配置

DJ 支持多种数据集输入类型，包括本地文件、远程数据集（如 huggingface）；还支持数据验证和数据混合。

配置输入文件的两种方法
- 简单场景，本地/HF 文件的单一路径
```yaml
dataset_path: '/path/to/your/dataset' # 数据集目录或文件的路径
```
- 高级方法，支持子配置项和更多功能
```yaml
dataset:
configs:
- type: 'local'
path: 'path/to/your/dataset' # 数据集目录或文件的路径
```

更多详细信息，请参阅 [数据集配置指南](docs/DatasetCfg_ZH.md)。

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

- **灵活的编程接口：**
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
 - RAY 模式下的去重算子与单机版本不同，所有 RAY 模式下的去重算子名称都以 `ray` 作为前缀，例如 `ray_video_deduplicator` 和 `ray_document_deduplicator`。
 - 更多细节请参考[分布式处理文档](docs/Distributed_ZH.md)。

> 用户也可以不使用 RAY，拆分数据集后使用 [Slurm](https://slurm.schedmd.com/) 在集群上运行，此时使用不包含 RAY 的原版 Data-Juicer 即可。
> [阿里云 PAI-DLC](https://www.aliyun.com/activity/bigdata/pai-dlc) 支持 RAY 框架、Slurm 框架等，用户可以直接在DLC集群上创建 RAY 作业 和 Slurm 作业。

### 数据分析

- 以配置文件路径为参数运行 `analyze_data.py` 或者 `dj-analyze` 命令行工具来分析数据集。

```shell
# 适用于从源码安装
python tools/analyze_data.py --config configs/demo/analyzer.yaml

# 使用命令行工具
dj-analyze --config configs/demo/analyzer.yaml

# 你也可以使用"自动"模式来避免写一个新的数据菜谱。它会使用全部可产出统计信息的 Filter 来分析
# 你的数据集的一小部分（如1000条样本，可通过 `auto_num` 参数指定）
dj-analyze --auto --dataset_path xx.jsonl [--auto_num 1000]
```

* **注意**：Analyzer 只用于能在 stats 字段里产出统计信息的 Filter 算子和能在 meta 字段里产出 tags 或类别标签的其他算子。除此之外的其他的算子会在分析过程中被忽略。我们使用以下两种注册器来装饰相关的算子：
  * `NON_STATS_FILTERS`：装饰那些**不能**产出任何统计信息的 Filter 算子。
  * `TAGGING_OPS`：装饰那些能在 meta 字段中产出 tags 或类别标签的算子。

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
  * ➕：从头开始构建自己的配置文件。您可以参考我们提供的样例配置文件 [`config_all.yaml`](configs/config_all.yaml)，[算子文档](docs/Operators.md)，以及 [开发者指南](docs/DeveloperGuide_ZH.md#构建自己的算子).
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

沙盒默认通过如下命令运行，更多介绍和细节请参阅[沙盒文档](docs/Sandbox_ZH.md).
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
  --privileged \
  --shm-size 256g \
  --network host \
  --gpus all \
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
  --privileged \
  --shm-size 256g \
  --network host \
  --gpus all \
  --rm \
  --name dj \
  -v <host_data_path>:<image_data_path> \
  -v ~/.cache/:/root/.cache/ \
  datajuicer/data-juicer:latest /bin/bash

# 进入这个容器，然后您可以在编辑模式下使用 data-juicer
docker exec -it <container_id> bash
```


<p align="right"><a href="#table">🔼 back to index</a></p>

## 开源协议

Data-Juicer 在 Apache License 2.0 协议下发布。

## 贡献

大模型是一个高速发展的领域，我们非常欢迎贡献新功能、修复漏洞以及文档改善。请参考[开发者指南](docs/DeveloperGuide_ZH.md)。


## 致谢

Data-Juicer被许多大模型相关产品和研究工作所使用，例如阿里巴巴通义和阿里云人工智能平台 (PAI) 之上的工业界场景。 我们期待更多您的体验反馈、建议和合作共建！


Data-Juicer 感谢社区[贡献者](https://github.com/modelscope/data-juicer/graphs/contributors) 和相关的先驱开源项目，譬如[Huggingface-Datasets](https://github.com/huggingface/datasets), [Bloom](https://huggingface.co/bigscience/bloom), [RedPajama](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1), [Arrow](https://github.com/apache/arrow), [Ray](https://github.com/ray-project/ray), ....

## 参考文献
如果您发现Data-Juicer对您的研发有帮助，请引用以下工作，[1.0paper](https://arxiv.org/abs/2309.02033), [2.0paper](https://arxiv.org/abs/2501.14755)。

```
@inproceedings{djv1,
  title={Data-Juicer: A One-Stop Data Processing System for Large Language Models},
  author={Daoyuan Chen and Yilun Huang and Zhijian Ma and Hesen Chen and Xuchen Pan and Ce Ge and Dawei Gao and Yuexiang Xie and Zhaoyang Liu and Jinyang Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
  booktitle={International Conference on Management of Data},
  year={2024}
}

@article{djv2,
  title={Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for Foundation Models},
  author={Chen, Daoyuan and Huang, Yilun and Pan, Xuchen and Jiang, Nana and Wang, Haibin and Ge, Ce and Chen, Yushuo and Zhang, Wenhao and Ma, Zhijian and Zhang, Yilei and Huang, Jun and Lin, Wei and Li, Yaliang and Ding, Bolin and Zhou, Jingren},
  journal={arXiv preprint arXiv:2501.14755},
  year={2024}
}
```
<details>
<summary>更多Data-Juicer团队关于数据的论文:
</summary>>

- [Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development](https://arxiv.org/abs/2407.11784)

- [ImgDiff: Contrastive Data Synthesis for Vision Large Language Models](https://arxiv.org/abs/2408.04594)

- [HumanVBench: Exploring Human-Centric Video Understanding Capabilities of MLLMs with Synthetic Benchmark Data](https://arxiv.org/abs/2412.17574)

- [The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective](https://arxiv.org/abs/2407.08583)

- [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://www.arxiv.org/abs/2502.04380)

- [MindGym: Enhancing Vision-Language Models via Synthetic Self-Challenging Questions](https://arxiv.org/abs/2503.09499)
  
- [BiMix: A Bivariate Data Mixing Law for Language Model Pretraining](https://arxiv.org/abs/2405.14908)

</details>



<p align="right"><a href="#table">🔼 back to index</a></p>