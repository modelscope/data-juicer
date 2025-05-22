
DJ-Cookbook
===========

Curated Resources
-----------------

* `KDD-Tutorial <https://modelscope.github.io/data-juicer/_static/tutorial_kdd24.html>`_
* :doc:`Awesome LLM-Data <docs/awesome_llm_data>`
* :doc:`"Bad" Data Exhibition <docs/BadDataExhibition>`

Coding with Data-Juicer (DJ)
----------------------------

* Basics

  * :doc:`Overview of DJ <index>`
  * :doc:`Quick Start <quick-start>`
  * :doc:`Configuration <docs/RecipeGallery>`
  * :doc:`Data Format Conversion <tools/fmt_conversion/README>`

* Lookup Materials

  * :doc:`DJ OperatorZoo <docs/Operators>`
  * :doc:`API references <api>`

* Advanced

  * :doc:`Developer Guide <docs/DeveloperGuide>`
  * :doc:`Preprocess Tools <tools/preprocess/README>`
  * :doc:`Postprocess Tools <tools/postprocess/README>`
  * :doc:`Sandbox <docs/Sandbox>`
  * :doc:`API Service <docs/DJ_service>`
  * :doc:`Data Scoring <tools/quality_classifier/README>`
  * :doc:`Auto Evaluation <tools/evaluator/README>`
  * :doc:`Third-parties Integration <thirdparty/LLM_ecosystems/README>`

Use Cases & Data Recipes
------------------------

* :doc:`Data Recipe Gallery <docs/RecipeGallery>`

  * Data-Juicer Minimal Example Recipe
  * Reproducing Open Source Text Datasets
  * Improving Open Source Pre-training Text Datasets
  * Improving Open Source Post-tuning Text Datasets
  * Synthetic Contrastive Learning Image-text Datasets
  * Improving Open Source Image-text Datasets
  * Basic Example Recipes for Video Data
  * Synthesizing Human-centric Video Benchmarks
  * Improving Existing Open Source Video Datasets

* Data-Juicer related Competitions

  * `Better Synth <https://tianchi.aliyun.com/competition/entrance/532251>`_\ , explore the impact of large model synthetic data on image understanding ability with DJ-Sandbox Lab and multimodal large models
  * `Modelscope-Sora Challenge <https://tianchi.aliyun.com/competition/entrance/532219>`_\ , based on Data-Juicer and `EasyAnimate <https://github.com/aigc-apps/EasyAnimate>`_ framework,  optimize data and train SORA-like small models to generate better videos
  * `Better Mixture <https://tianchi.aliyun.com/competition/entrance/532174>`_\ , only adjust data mixing and sampling strategies for given multiple candidate datasets
  * FT-Data Ranker (`1B Track <https://tianchi.aliyun.com/competition/entrance/532157>`_\ , `7B Track <https://tianchi.aliyun.com/competition/entrance/532158>`_\ ), For a specified candidate dataset, only adjust the data filtering and enhancement strategies
  * `Kolors-LoRA Stylized Story Challenge <https://tianchi.aliyun.com/competition/entrance/532254>`_\ , based on Data-Juicer and `DiffSynth-Studio <https://github.com/modelscope/DiffSynth-Studio>`_\ framework, explore Diffusion model fine-tuning

* :doc:`DJ-SORA <docs/DJ_SORA>`
* Based on Data-Juicer and `AgentScope <https://github.com/modelscope/agentscope>`_ framework, leverage `agents to call DJ Filters <https://github.com/modelscope/data-juicer/blob/main/demos/api_service/react_data_filter_process.ipynb>` and `call DJ Mappers <https://github.com/modelscope/data-juicer/blob/main/demos/api_service/react_data_mapper_process.ipynb>`

Interactive Examples
--------------------

* Introduction to Data-Juicer [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/overview_scan/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/overview_scan>`_\ ]
* Data Visualization:

  * Basic Statistics [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_visualization_statistics>`_\ ]
  * Lexical Diversity [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_visualization_diversity>`_\ ]
  * Operator Insight (Single OP) [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_visualization_op_insight/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_visualization_op_insight>`_\ ]
  * Operator Effect (Multiple OPs) [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_visualization_op_effect>`_\ ]

* Data Processing:

  * Scientific Literature (e.g. `arXiv <https://info.arxiv.org/help/bulk_data_s3.html>`_\ ) [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/process_sci_data>`_\ ]
  * Programming Code (e.g. `TheStack <https://huggingface.co/datasets/bigcode/the-stack>`_\ ) [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/process_code_data/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/process_code_data>`_\ ]
  * Chinese Instruction Data (e.g. `Alpaca-CoT <https://huggingface.co/datasets/QingyiSi/Alpaca-CoT>`_\ ) [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/process_cft_zh_data>`_\ ]

* Tool Pool:

  * Dataset Splitting by Language [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language>`_\ ]
  * Quality Classifier for CommonCrawl [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/tool_quality_classifier>`_\ ]
  * Auto Evaluation on `HELM <https://github.com/stanford-crfm/helm>` [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/auto_evaluation_helm>`_\ ]
  * Data Sampling and Mixture [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_mixture/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_mixture>`_\ ]

* Data Processing Loop [\ `ModelScope <https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary>`_\ ] [\ `HuggingFace <https://huggingface.co/spaces/datajuicer/data_process_loop>`_\ ]
