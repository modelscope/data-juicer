DJ-Cookbook
-----------

资源合集
^^^^^^^^


* `KDD'24 相关教程 <https://modelscope.github.io/data-juicer/_static/tutorial_kdd24.html>`_
* `Awesome LLM-Data <docs/awesome_llm_data>`_
* `“坏”数据展览 <docs/BadDataExhibition_ZH>`_

编写Data-Juicer (DJ) 代码
^^^^^^^^^^^^^^^^^^^^^^^^^


* 基础

  * `DJ概览 <index_ZH>`_
  * `快速上手 <quick-start_ZH>`_
  * `配置 <docs/RecipeGallery_ZH>`_
  * `数据格式转换 <tools/fmt_conversion/README_ZH>`_

* 信息速查

  * `算子库 <docs/Operators>`_
  * `API参考 <https://modelscope.github.io/data-juicer/>`_

* 进阶

  * `开发者指南 <docs/DeveloperGuide_ZH>`_
  * `预处理工具 <tools/preprocess/README_ZH>`_
  * `后处理工具 <tools/postprocess/README_ZH>`_
  * `沙盒 <docs/Sandbox_ZH>`_
  * `API服务化 <docs/DJ_service_ZH>`_
  * `给数据打分 <tools/quality_classifier/README_ZH>`_
  * `自动评估 <tools/evaluator/README_ZH>`_
  * `第三方集成 <thirdparty/LLM_ecosystems/README_ZH>`_

用例与数据菜谱
^^^^^^^^^^^^^^


* `数据菜谱Gallery <docs/RecipeGallery_ZH>`_

  * Data-Juicer 最小示例配方
  * 复现开源文本数据集
  * 改进开源文本预训练数据集
  * 改进开源文本后处理数据集
  * 合成对比学习图像文本数据集
  * 改进开源图像文本数据集
  * 视频数据的基本示例菜谱
  * 合成以人为中心的视频评测集
  * 改进现有的开源视频数据集

* Data-Juicer相关竞赛

  * `Better Synth <https://tianchi.aliyun.com/competition/entrance/532251>`_\ ，在DJ-沙盒实验室和多模态大模型上，探索大模型合成数据对图像理解能力的影响
  * `Modelscope-Sora挑战赛 <https://tianchi.aliyun.com/competition/entrance/532219>`_\ ，基于Data-Juicer和\ `EasyAnimate <https://github.com/aigc-apps/EasyAnimate>`_\ 框架，调优文本-视频数据集，在类SORA小模型上训练以生成更好的视频
  * `Better Mixture <https://tianchi.aliyun.com/competition/entrance/532174>`_\ ，针对指定多个候选数据集，仅调整数据混合和采样策略
  * FT-Data Ranker (\ `1B Track <https://tianchi.aliyun.com/competition/entrance/532157>`_\ 、 `7B Track <https://tianchi.aliyun.com/competition/entrance/532158>`_\ )，针对指定候选数据集，仅调整数据过滤和增强策略
  * `可图Kolors-LoRA风格故事挑战赛 <https://tianchi.aliyun.com/competition/entrance/532254>`_\ ，基于Data-Juicer和\ `DiffSynth-Studio <https://github.com/modelscope/DiffSynth-Studio>`_\ 框架，探索Diffusion模型微调

* `DJ-SORA <docs/DJ_SORA_ZH>`_
* 基于Data-Juicer和\ `AgentScope <https://github.com/modelscope/agentscope>`_\ 框架，通过\ `智能体调用DJ Filters <https://github.com/modelscope/data-juicer/blob/main/demos/api_service/react_data_filter_process.ipynb>`_\ 和\ `调用DJ Mappers <https://github.com/modelscope/data-juicer/blob/main/demos/api_service/react_data_mapper_process.ipynb>`_

交互类示例
^^^^^^^^^^


* Data-Juicer 介绍 [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/overview_scan/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/overview_scan>`_\ ]
* 数据可视化:

  * 基础指标统计 [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_visualization_statistics>`_\ ]
  * 词汇多样性 [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_visualization_diversity>`_\ ]
  * 算子洞察（单OP） [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_visualization_op_insight/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_visualization_op_insight>`_\ ]
  * 算子效果（多OP） [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_visualization_op_effect>`_\ ]

* 数据处理:

  * 科学文献 (例如 `arXiv <https://info.arxiv.org/help/bulk_data_s3.html>`_\ ) [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/process_sci_data>`_\ ]
  * 编程代码 (例如 `TheStack <https://huggingface.co/datasets/bigcode/the-stack>`_\ ) [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/process_code_data/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/process_code_data>`_\ ]
  * 中文指令数据 (例如 `Alpaca-CoT <https://huggingface.co/datasets/QingyiSi/Alpaca-CoT>`_\ ) [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/process_cft_zh_data>`_\ ]

* 工具池:

  * 按语言分割数据集 [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language>`_\ ]
  * CommonCrawl 质量分类器 [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/tool_quality_classifier>`_\ ]
  * 基于 `HELM <https://github.com/stanford-crfm/helm>`_ 的自动评测 [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/auto_evaluation_helm>`_\ ]
  * 数据采样及混合 [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_mixture/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_mixture>`_\ ]

* 数据处理回路 [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_process_loop>`_\ ]
