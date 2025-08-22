
# Operator Schemas 算子提要

Operators are a collection of basic processes that assist in data modification,
cleaning, filtering, deduplication, etc. We support a wide range of data
sources and file formats, and allow for flexible extension to custom datasets.

算子 (Operator) 是协助数据修改、清理、过滤、去重等基本流程的集合。我们支持广泛的数据来源和文件格式，并支持对自定义数据集的灵活扩展。

This page offers a basic description of the operators (OPs) in Data-Juicer.
Users can refer to the
[API documentation](https://modelscope.github.io/data-juicer/) for the specific
parameters of each operator. Users can refer to and run the unit tests
(`tests/ops/...`) for [examples of operator-wise usage](../tests/ops) as well
as the effects of each operator when applied to built-in test data samples.
Besides, you can try to use agent to automatically route suitable OPs and
call them. E.g., refer to
[Agentic Filters of DJ](../demos/api_service/react_data_filter_process.ipynb),
 [Agentic Mappers of DJ](../demos/api_service/react_data_mapper_process.ipynb)

这个页面提供了OP的基本描述，用户可以参考[API文档](https://modelscope.github.io/data-juicer/)更细致了解每个
OP的具体参数，并且可以查看、运行单元测试 (`tests/ops/...`)，来体验
[各OP的用法示例](../tests/ops)以及每个OP作用于内置测试数据样本时的效果。例如，参考
[Agentic Filters of DJ](../demos/api_service/react_data_filter_process.ipynb),
 [Agentic Mappers of DJ](../demos/api_service/react_data_mapper_process.ipynb)


## Overview  概览

The operators in Data-Juicer are categorized into 7 types.
Data-Juicer 中的算子分为以下 7 种类型。

| Type 类型 | Number 数量 | Description 描述 |
|------|:------:|-------------|
| [aggregator](#aggregator) | 4 | Aggregate for batched samples, such as summary or conclusion. 对批量样本进行汇总，如得出总结或结论。 |
| [deduplicator](#deduplicator) | 10 | Detects and removes duplicate samples. 识别、删除重复样本。 |
| [filter](#filter) | 54 | Filters out low-quality samples. 过滤低质量样本。 |
| [formatter](#formatter) | 8 | Discovers, loads, and canonicalizes source data. 发现、加载、规范化原始数据。 |
| [grouper](#grouper) | 3 | Group samples to batched samples. 将样本分组，每一组组成一个批量样本。 |
| [mapper](#mapper) | 81 | Edits and transforms samples. 对数据样本进行编辑和转换。 |
| [selector](#selector) | 5 | Selects top samples based on ranking. 基于排序选取高质量样本。 |

All the specific operators are listed below, each featured with several capability tags. 
下面列出所有具体算子，每种算子都通过多个标签来注明其主要功能。
* Modality Tags
  - 🔤Text: process text data specifically. 专用于处理文本。
  - 🏞Image: process image data specifically. 专用于处理图像。
  - 📣Audio: process audio data specifically. 专用于处理音频。
  - 🎬Video: process video data specifically. 专用于处理视频。
  - 🔮Multimodal: process multimodal data. 用于处理多模态数据。
* Resource Tags
  - 💻CPU: only requires CPU resource. 只需要 CPU 资源。
  - 🚀GPU: requires GPU/CUDA resource as well. 额外需要 GPU/CUDA 资源。
* Usability Tags
  - 🔴Alpha: alpha version OP. Only the basic OP implementations are finished. 表示 alpha 版本算子。只完成了基础的算子实现。
  - 🟡Beta: beta version OP. Based on the alpha version, unittests for this OP are added as well. 表示 beta 版本算子。基于 alpha 版本，添加了算子的单元测试。
  - 🟢Stable: stable version OP. Based on the beta version, OP optimizations related to DJ (e.g. model management, batched processing, OP fusion, ...) are added to this OP. 表示 stable 版本算子。基于 beta 版本，完善了DJ相关的算子优化项（如模型管理，批处理，算子融合等）。
* Model Tags
  - 🔗API: equipped with API-based models. (e.g. ChatGPT, GPT-4o). 支持基于 API 调用模型（如 ChatGPT，GPT-4o）。
  - 🌊vLLM: equipped with models supported by vLLM. 支持基于 vLLM 进行模型推理。
  - 🧩HF: equipped with models from HuggingFace Hub. 支持来自于 HuggingFace Hub 的模型。

## aggregator <a name="aggregator"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| entity_attribute_aggregator | 💻CPU 🔗API 🟢Stable | Return conclusion of the given entity's attribute from some docs. 从一些文档返回给定实体的属性的结论。 | - | - |
| meta_tags_aggregator | 💻CPU 🔗API 🟢Stable | Merge similar meta tags to one tag. 将类似的元标记合并到一个标记。 | - | - |
| most_relevant_entities_aggregator | 💻CPU 🔗API 🟢Stable | Extract entities closely related to a given entity from some texts, and sort them in descending order of importance. 从一些文本中提取与给定实体密切相关的实体，并按重要性的降序对它们进行排序。 | - | - |
| nested_aggregator | 🔤Text 💻CPU 🔗API 🟢Stable | Considering the limitation of input length, nested aggregate contents for each given number of samples. 考虑到输入长度的限制，嵌套聚合每个给定数量的样本的内容。 | - | - |

## deduplicator <a name="deduplicator"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| document_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using exact matching. Deduplicator使用精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/document_deduplicator.md) | - |
| document_minhash_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using MinHashLSH. Deduplicator使用MinHashLSH在文档级别删除重复的样本。 | [info](operators/deduplicator/document_minhash_deduplicator.md) | - |
| document_simhash_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using SimHash. Deduplicator使用SimHash在文档级别对样本进行重复数据删除。 | [info](operators/deduplicator/document_simhash_deduplicator.md) | - |
| image_deduplicator | 🏞Image 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using exact matching of images between documents. Deduplicator使用文档之间的图像精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/image_deduplicator.md) | - |
| ray_basic_deduplicator | 💻CPU 🔴Alpha | Backend for deduplicator. deduplicator的后端。 | - | - |
| ray_bts_minhash_deduplicator | 🔤Text 💻CPU 🟡Beta | A distributed implementation of Union-Find with load balancing. 具有负载平衡的Union-Find的分布式实现。 | [info](operators/deduplicator/ray_bts_minhash_deduplicator.md) | - |
| ray_document_deduplicator | 🔤Text 💻CPU 🟡Beta | Deduplicator to deduplicate samples at document-level using exact matching. Deduplicator使用精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/ray_document_deduplicator.md) | - |
| ray_image_deduplicator | 🏞Image 💻CPU 🟡Beta | Deduplicator to deduplicate samples at document-level using exact matching of images between documents. Deduplicator使用文档之间的图像精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/ray_image_deduplicator.md) | - |
| ray_video_deduplicator | 🎬Video 💻CPU 🟡Beta | Deduplicator to deduplicate samples at document-level using exact matching of videos between documents. Deduplicator使用文档之间的视频精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/ray_video_deduplicator.md) | - |
| video_deduplicator | 🎬Video 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using exact matching of videos between documents. Deduplicator使用文档之间的视频精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/video_deduplicator.md) | - |

## filter <a name="filter"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| alphanumeric_filter | 🔤Text 💻CPU 🧩HF 🟢Stable | Filter to keep samples with alphabet/numeric ratio within a specific range. 过滤器保持样品与字母/数字的比例在一个特定的范围内。 | [info](operators/filter/alphanumeric_filter.md) | - |
| audio_duration_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples whose audios' durations are within a specified range. 保留音频持续时间在指定范围内的数据样本。 | [info](operators/filter/audio_duration_filter.md) | - |
| audio_nmf_snr_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples whose audios' SNRs (computed based on NMF) are within a specified range. 保留音频的snr (根据NMF计算) 在指定范围内的数据样本。 | [info](operators/filter/audio_nmf_snr_filter.md) | - |
| audio_size_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples whose audio size (in bytes/kb/MB/...) within a specific range. 保留音频大小 (以字节/kb/MB/... 为单位) 在特定范围内的数据样本。 | [info](operators/filter/audio_size_filter.md) | - |
| average_line_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with average line length within a specific range. 过滤器，以保持平均线长度在特定范围内的样本。 | - | - |
| character_repetition_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with char-level n-gram repetition ratio within a specific range. 过滤器将具有char级n-gram重复比率的样本保持在特定范围内。 | [info](operators/filter/character_repetition_filter.md) | - |
| flagged_words_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with flagged-word ratio less than a specific max value. 过滤以保持标记词比率小于特定最大值的样本。 | [info](operators/filter/flagged_words_filter.md) | - |
| general_field_filter | 💻CPU 🟡Beta | Filter to keep samples based on a general field filter condition. 根据常规字段筛选条件保留样本。 | - | - |
| image_aesthetics_filter | 🏞Image 🚀GPU 🧩HF 🟢Stable | Filter to keep samples with aesthetics scores within a specific range. 过滤以保持美学分数在特定范围内的样品。 | [info](operators/filter/image_aesthetics_filter.md) | - |
| image_aspect_ratio_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with image aspect ratio within a specific range. 过滤器，以保持样本的图像纵横比在特定范围内。 | [info](operators/filter/image_aspect_ratio_filter.md) | - |
| image_face_count_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with the number of faces within a specific range. 过滤以保持样本的面数在特定范围内。 | [info](operators/filter/image_face_count_filter.md) | - |
| image_face_ratio_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with face area ratios within a specific range. 过滤以保持面面积比在特定范围内的样本。 | [info](operators/filter/image_face_ratio_filter.md) | - |
| image_nsfw_filter | 🏞Image 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose images have low nsfw scores. 过滤器保留图像具有低nsfw分数的样本。 | [info](operators/filter/image_nsfw_filter.md) | - |
| image_pair_similarity_filter | 🏞Image 🚀GPU 🧩HF 🟢Stable | Filter to keep image pairs with similarities between images within a specific range. 过滤器将图像之间具有相似性的图像对保持在特定范围内。 | - | - |
| image_shape_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with image shape (w, h) within specific ranges. 过滤器保持样品的图像形状 (w，h) 在特定范围内。 | [info](operators/filter/image_shape_filter.md) | - |
| image_size_filter | 🏞Image 💻CPU 🟢Stable | Keep data samples whose image size (in Bytes/KB/MB/...) within a specific range. 保留图像大小 (以字节/KB/MB/... 为单位) 在特定范围内的数据样本。 | [info](operators/filter/image_size_filter.md) | - |
| image_text_matching_filter | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Filter to keep samples those matching score between image and text within a specific range. 过滤器将图像和文本之间的匹配分数保持在特定范围内。 | [info](operators/filter/image_text_matching_filter.md) | - |
| image_text_similarity_filter | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Filter to keep samples those similarities between image and text within a specific range. 过滤器将图像和文本之间的相似性保持在特定范围内。 | [info](operators/filter/image_text_similarity_filter.md) | - |
| image_watermark_filter | 🏞Image 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose images have no watermark with high probability. 过滤器，以保留图像没有水印的样本。 | [info](operators/filter/image_watermark_filter.md) | - |
| in_context_influence_filter | 🚀GPU 🟢Stable | Filter to keep texts whose in-context influence upon validation set within a specific range. 过滤器，以将上下文对验证的影响设置在特定范围内的文本保留下来。 | [info](operators/filter/in_context_influence_filter.md) | - |
| instruction_following_difficulty_filter | 🚀GPU 🟡Beta | Filter to keep texts whose instruction follows difficulty (IFD, https://arxiv.org/abs/2308.12032) falls within a specific range. 过滤以保持其指令跟随难度 (IFD， https://arxiv.org/abs/ 2308.12032) 落在特定范围内的文本。 | [info](operators/filter/instruction_following_difficulty_filter.md) | - |
| language_id_score_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples in a specific language with confidence score larger than a specific min value. 过滤器以保留置信度得分大于特定最小值的特定语言的样本。 | [info](operators/filter/language_id_score_filter.md) | - |
| llm_analysis_filter | 🚀GPU 🌊vLLM 🧩HF 🔗API 🟡Beta | Base filter class for leveraging LLMs to filter various samples. 用于利用llm过滤各种样本的基本筛选器类。 | - | - |
| llm_difficulty_score_filter | 💻CPU 🟡Beta | Filter to keep sample with high difficulty score estimated by LLM. 过滤器以保持LLM估计的高难度分数的样本。 | - | - |
| llm_perplexity_filter | 🚀GPU 🧩HF 🟡Beta | Filter to keep samples with perplexity score, computed using a specified llm, within a specific range. 过滤器将使用指定llm计算的具有困惑度分数的样本保持在特定范围内。 | [info](operators/filter/llm_perplexity_filter.md) | - |
| llm_quality_score_filter | 💻CPU 🟡Beta | Filter to keep sample with high quality score estimated by LLM. 过滤器以保持LLM估计的高质量分数的样本。 | - | - |
| llm_task_relevance_filter | 💻CPU 🟡Beta | Filter to keep sample with high relevance score to validation tasks estimated by LLM. 过滤以保持与LLM估计的验证任务具有高相关性得分的样本。 | [info](operators/filter/llm_task_relevance_filter.md) | - |
| maximum_line_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with maximum line length within a specific range. 过滤器将最大行长度的样本保持在特定范围内。 | - | - |
| perplexity_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with perplexity score less than a specific max value. 过滤以保留困惑度分数小于特定最大值的样本。 | [info](operators/filter/perplexity_filter.md) | - |
| phrase_grounding_recall_filter | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose locating recalls of phrases extracted from text in the images are within a specified range. 过滤器，用于保留从图像中的文本中提取的短语的定位回忆在指定范围内的样本。 | [info](operators/filter/phrase_grounding_recall_filter.md) | - |
| special_characters_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with special-char ratio within a specific range. 过滤器，以保持样品与特殊字符的比例在一个特定的范围内。 | [info](operators/filter/special_characters_filter.md) | - |
| specified_field_filter | 💻CPU 🟢Stable | Filter based on specified field information. 根据指定的字段信息进行筛选。 | [info](operators/filter/specified_field_filter.md) | - |
| specified_numeric_field_filter | 💻CPU 🟢Stable | Filter based on specified numeric field information. 根据指定的数值字段信息进行筛选。 | [info](operators/filter/specified_numeric_field_filter.md) | - |
| stopwords_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with stopword ratio larger than a specific min value. 过滤以保持停止词比率大于特定最小值的样本。 | [info](operators/filter/stopwords_filter.md) | - |
| suffix_filter | 💻CPU 🟢Stable | Filter to keep samples with specified suffix. 过滤器以保留具有指定后缀的样本。 | - | - |
| text_action_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep texts those contain actions in the text. 过滤以保留文本中包含操作的文本。 | [info](operators/filter/text_action_filter.md) | - |
| text_embd_similarity_filter | 🔤Text 🚀GPU 🔗API 🟡Beta | Filter to keep texts whose average embedding similarity to a set of given validation texts falls within a specific range. 过滤器，以保留与一组给定验证文本的平均嵌入相似度在特定范围内的文本。 | [info](operators/filter/text_embd_similarity_filter.md) | - |
| text_entity_dependency_filter | 🔤Text 💻CPU 🟢Stable | Identify the entities in the text which are independent with other token, and filter them. 识别文本中与其他令牌独立的实体，并对其进行过滤。 | [info](operators/filter/text_entity_dependency_filter.md) | - |
| text_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with total text length within a specific range. 过滤以保持文本总长度在特定范围内的样本。 | [info](operators/filter/text_length_filter.md) | - |
| text_pair_similarity_filter | 🔤Text 🚀GPU 🧩HF 🟢Stable | Filter to keep text pairs with similarities between texts within a specific range. 过滤器将文本之间具有相似性的文本对保留在特定范围内。 | [info](operators/filter/text_pair_similarity_filter.md) | - |
| token_num_filter | 🔤Text 💻CPU 🧩HF 🟢Stable | Filter to keep samples with total token number within a specific range. 筛选器将总令牌数的样本保留在特定范围内。 | - | - |
| video_aesthetics_filter | 🎬Video 🚀GPU 🧩HF 🟢Stable | Filter to keep data samples with aesthetics scores for specified frames in the videos within a specific range. 过滤器将视频中指定帧的美学得分数据样本保留在特定范围内。 | - | - |
| video_aspect_ratio_filter | 🎬Video 💻CPU 🟢Stable | Filter to keep samples with video aspect ratio within a specific range. 过滤器将视频纵横比的样本保持在特定范围内。 | [info](operators/filter/video_aspect_ratio_filter.md) | - |
| video_duration_filter | 🎬Video 💻CPU 🟢Stable | Keep data samples whose videos' durations are within a specified range. 保留视频持续时间在指定范围内的数据样本。 | [info](operators/filter/video_duration_filter.md) | - |
| video_frames_text_similarity_filter | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Filter to keep samples those similarities between sampled video frame images and text within a specific range. 过滤以保持采样视频帧图像和文本之间的相似性在特定范围内。 | [info](operators/filter/video_frames_text_similarity_filter.md) | - |
| video_motion_score_filter | 🎬Video 💻CPU 🟢Stable | Filter to keep samples with video motion scores within a specific range. 过滤器将视频运动分数的样本保持在特定范围内。 | [info](operators/filter/video_motion_score_filter.md) | - |
| video_motion_score_raft_filter | 🎬Video 🚀GPU 🟢Stable | Filter to keep samples with video motion scores within a specified range. 过滤器将视频运动分数的样本保持在指定范围内。 | [info](operators/filter/video_motion_score_raft_filter.md) | - |
| video_nsfw_filter | 🎬Video 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose videos have low nsfw scores. 过滤器以保留其视频具有低nsfw分数的样本。 | [info](operators/filter/video_nsfw_filter.md) | - |
| video_ocr_area_ratio_filter | 🎬Video 🚀GPU 🟢Stable | Keep data samples whose detected text area ratios for specified frames in the video are within a specified range. 保留检测到的视频中指定帧的文本面积比率在指定范围内的数据样本。 | [info](operators/filter/video_ocr_area_ratio_filter.md) | - |
| video_resolution_filter | 🎬Video 💻CPU 🟢Stable | Keep data samples whose videos' resolutions are within a specified range. 保留视频分辨率在指定范围内的数据样本。 | [info](operators/filter/video_resolution_filter.md) | - |
| video_tagging_from_frames_filter | 🎬Video 🚀GPU 🟢Stable | Filter to keep samples whose videos contain the given tags. 过滤器以保留其视频包含给定标签的样本。 | [info](operators/filter/video_tagging_from_frames_filter.md) | - |
| video_watermark_filter | 🎬Video 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose videos have no watermark with high probability. 过滤器以保持其视频具有高概率没有水印的样本。 | [info](operators/filter/video_watermark_filter.md) | - |
| word_repetition_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with word-level n-gram repetition ratio within a specific range. 过滤器将单词级n-gram重复比率的样本保持在特定范围内。 | [info](operators/filter/word_repetition_filter.md) | - |
| words_num_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with total words number within a specific range. 过滤器，以保持总字数在特定范围内的样本。 | [info](operators/filter/words_num_filter.md) | - |

## formatter <a name="formatter"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| csv_formatter | 🟢Stable | The class is used to load and format csv-type files. 类用于加载和格式化csv类型的文件。 | - | - |
| empty_formatter | 🟢Stable | The class is used to create empty data. 类用于创建空数据。 | - | - |
| json_formatter | 🟡Beta | The class is used to load and format json-type files. 类用于加载和格式化json类型的文件。 | - | - |
| local_formatter | 🟢Stable | The class is used to load a dataset from local files or local directory. 类用于从本地文件或本地目录加载数据集。 | - | - |
| parquet_formatter | 🟢Stable | The class is used to load and format parquet-type files. 该类用于加载和格式化镶木地板类型的文件。 | - | - |
| remote_formatter | 🟢Stable | The class is used to load a dataset from repository of huggingface hub. 该类用于从huggingface hub的存储库加载数据集。 | - | - |
| text_formatter | 🔴Alpha | The class is used to load and format text-type files. 类用于加载和格式化文本类型文件。 | - | - |
| tsv_formatter | 🟢Stable | The class is used to load and format tsv-type files. 该类用于加载和格式化tsv类型的文件。 | - | - |

## grouper <a name="grouper"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| key_value_grouper | 🔤Text 💻CPU 🟢Stable | Group samples to batched samples according values in given keys. 根据给定键中的值将样本分组为批处理样本。 | - | - |
| naive_grouper | 💻CPU 🟢Stable | Group all samples to one batched sample. 将所有样品分组为一批样品。 | - | - |
| naive_reverse_grouper | 💻CPU 🟢Stable | Split batched samples to samples. 将批处理的样品拆分为样品。 | - | - |

## mapper <a name="mapper"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| audio_add_gaussian_noise_mapper | 📣Audio 💻CPU 🟡Beta | Mapper to add gaussian noise to audio. 映射器向音频添加高斯噪声。 | - | - |
| audio_ffmpeg_wrapped_mapper | 📣Audio 💻CPU 🟢Stable | Simple wrapper for FFmpeg audio filters. FFmpeg音频滤波器的简单包装。 | [info](operators/mapper/audio_ffmpeg_wrapped_mapper.md) | - |
| calibrate_qa_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Mapper to calibrate question-answer pairs based on reference text. 映射器基于参考文本校准问答对。 | - | - |
| calibrate_query_mapper | 💻CPU 🟢Stable | Mapper to calibrate query in question-answer pairs based on reference text. 映射器基于参考文本校准问答对中的查询。 | - | - |
| calibrate_response_mapper | 💻CPU 🟢Stable | Mapper to calibrate response in question-answer pairs based on reference text. 映射器基于参考文本校准问答对中的响应。 | - | - |
| chinese_convert_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to convert Chinese between Traditional Chinese, Simplified Chinese and Japanese Kanji. 映射器在繁体中文，简体中文和日语汉字之间转换中文。 | [info](operators/mapper/chinese_convert_mapper.md) | - |
| clean_copyright_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean copyright comments at the beginning of the text samples. Mapper清理版权注释开头的文本样本。 | [info](operators/mapper/clean_copyright_mapper.md) | - |
| clean_email_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean email in text samples. 映射器清理文本样本中的电子邮件。 | [info](operators/mapper/clean_email_mapper.md) | - |
| clean_html_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean html code in text samples. 映射器来清理文本示例中的html代码。 | [info](operators/mapper/clean_html_mapper.md) | - |
| clean_ip_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean ipv4 and ipv6 address in text samples. 映射器以清除文本示例中的ipv4和ipv6地址。 | [info](operators/mapper/clean_ip_mapper.md) | - |
| clean_links_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean links like http/https/ftp in text samples. 映射器来清理链接，如文本示例中的http/https/ftp。 | [info](operators/mapper/clean_links_mapper.md) | - |
| dialog_intent_detection_mapper | 💻CPU 🔗API 🟢Stable | Mapper to generate user's intent labels in dialog. 映射器在对话框中生成用户的意图标签。 | [info](operators/mapper/dialog_intent_detection_mapper.md) | - |
| dialog_sentiment_detection_mapper | 💻CPU 🔗API 🟢Stable | Mapper to generate user's sentiment labels in dialog. 映射器在对话框中生成用户的情绪标签。 | [info](operators/mapper/dialog_sentiment_detection_mapper.md) | - |
| dialog_sentiment_intensity_mapper | 💻CPU 🔗API 🟢Stable | Mapper to predict user's sentiment intensity (from -5 to 5 in default prompt) in dialog. Mapper在对话框中预测用户的情绪强度 (在默认提示中从-5到5)。 | [info](operators/mapper/dialog_sentiment_intensity_mapper.md) | - |
| dialog_topic_detection_mapper | 💻CPU 🔗API 🟢Stable | Mapper to generate user's topic labels in dialog. 映射器在对话框中生成用户的主题标签。 | [info](operators/mapper/dialog_topic_detection_mapper.md) | - |
| download_file_mapper | 💻CPU 🟡Beta | Mapper to download url files to local files or load them into memory. 映射器将url文件下载到本地文件或将其加载到内存中。 | - | - |
| expand_macro_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to expand macro definitions in the document body of Latex samples. Mapper来扩展Latex示例文档主体中的宏定义。 | [info](operators/mapper/expand_macro_mapper.md) | - |
| extract_entity_attribute_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract attributes for given entities from the text. 从文本中提取给定实体的属性。 | - | - |
| extract_entity_relation_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract entities and relations in the text for knowledge graph. 提取知识图谱的文本中的实体和关系。 | - | - |
| extract_event_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract events and relevant characters in the text. 提取文本中的事件和相关字符。 | - | - |
| extract_keyword_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Generate keywords for the text. 为文本生成关键字。 | - | - |
| extract_nickname_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract nickname relationship in the text. 提取文本中的昵称关系。 | - | - |
| extract_support_text_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract support sub text for a summary. 提取摘要的支持子文本。 | - | - |
| extract_tables_from_html_mapper | 🔤Text 💻CPU 🟡Beta | Mapper to extract tables from HTML content. 映射器从HTML内容中提取表。 | - | - |
| fix_unicode_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to fix unicode errors in text samples. 映射器修复文本示例中的unicode错误。 | [info](operators/mapper/fix_unicode_mapper.md) | - |
| generate_qa_from_examples_mapper | 🚀GPU 🌊vLLM 🧩HF 🟢Stable | Mapper to generate question and answer pairs from examples. 映射器从示例生成问题和答案对。 | - | - |
| generate_qa_from_text_mapper | 🔤Text 🚀GPU 🌊vLLM 🧩HF 🟢Stable | Mapper to generate question and answer pairs from text. 映射器从文本生成问题和答案对。 | [info](operators/mapper/generate_qa_from_text_mapper.md) | - |
| image_blur_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to blur images. 映射器来模糊图像。 | - | - |
| image_captioning_from_gpt4v_mapper | 🔮Multimodal 💻CPU 🟡Beta | Mapper to generate samples whose texts are generated based on gpt-4-vision and the image. Mapper生成样本，其文本基于gpt-4-vision和图像生成。 | - | - |
| image_captioning_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to generate samples whose captions are generated based on another model and the figure. 映射器生成样本，其标题是基于另一个模型和图生成的。 | - | - |
| image_diffusion_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Generate image by diffusion model. 通过扩散模型生成图像。 | - | - |
| image_face_blur_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to blur faces detected in images. 映射器模糊图像中检测到的人脸。 | - | - |
| image_remove_background_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to remove background of images. 映射器删除图像的背景。 | - | - |
| image_segment_mapper | 🏞Image 🚀GPU 🟢Stable | Perform segment-anything on images and return the bounding boxes. 对图像执行segment-任何操作并返回边界框。 | - | - |
| image_tagging_mapper | 🏞Image 🚀GPU 🟢Stable | Mapper to generate image tags. 映射器生成图像标签。 | - | - |
| imgdiff_difference_area_generator_mapper | 🚀GPU 🟡Beta | A fused operator for OPs that is used to run sequential OPs on the same batch to allow fine-grained control on data processing. OPs的融合操作符，用于在同一批次上运行顺序OPs，以实现对数据处理的细粒度控制。 | - | - |
| imgdiff_difference_caption_generator_mapper | 🚀GPU 🟡Beta | A fused operator for OPs that is used to run sequential OPs on the same batch to allow fine-grained control on data processing. OPs的融合操作符，用于在同一批次上运行顺序OPs，以实现对数据处理的细粒度控制。 | - | - |
| mllm_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to use MLLMs for visual question answering tasks. Mapper使用MLLMs进行视觉问答任务。 | - | - |
| nlpaug_en_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to simply augment samples in English based on nlpaug library. 映射器基于nlpaug库简单地增加英语样本。 | - | - |
| nlpcda_zh_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to simply augment samples in Chinese based on nlpcda library. 基于nlpcda库的映射器可以简单地增加中文样本。 | - | - |
| optimize_qa_mapper | 🚀GPU 🌊vLLM 🧩HF 🟢Stable | Mapper to optimize question-answer pairs. 映射器来优化问题-答案对。 | [info](operators/mapper/optimize_qa_mapper.md) | - |
| optimize_query_mapper | 🚀GPU 🟢Stable | Mapper to optimize query in question-answer pairs. 映射器来优化问答对中的查询。 | [info](operators/mapper/optimize_query_mapper.md) | - |
| optimize_response_mapper | 🚀GPU 🟢Stable | Mapper to optimize response in question-answer pairs. 映射器来优化问答对中的响应。 | [info](operators/mapper/optimize_response_mapper.md) | - |
| pair_preference_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Mapper to construct paired preference samples. 映射器来构造成对的偏好样本。 | [info](operators/mapper/pair_preference_mapper.md) | - |
| punctuation_normalization_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to normalize unicode punctuations to English punctuations in text samples. 映射器将文本示例中的unicode标点规范化为英文标点。 | [info](operators/mapper/punctuation_normalization_mapper.md) | - |
| python_file_mapper | 💻CPU 🟢Stable | Mapper for executing Python function defined in a file. Mapper用于执行文件中定义的Python函数。 | - | - |
| python_lambda_mapper | 💻CPU 🟢Stable | Mapper for executing Python lambda function on data samples. 用于对数据示例执行Python lambda函数的映射器。 | - | - |
| query_intent_detection_mapper | 🚀GPU 🧩HF 🧩HF 🟢Stable | Mapper to predict user's Intent label in query. Mapper在查询中预测用户的意图标签。 | [info](operators/mapper/query_intent_detection_mapper.md) | - |
| query_sentiment_detection_mapper | 🚀GPU 🧩HF 🧩HF 🟢Stable | Mapper to predict user's sentiment label ('negative', 'neutral' and 'positive') in query. Mapper在查询中预测用户的情绪标签 (“负面”，“中性” 和 “正面”)。 | [info](operators/mapper/query_sentiment_detection_mapper.md) | - |
| query_topic_detection_mapper | 🚀GPU 🧩HF 🧩HF 🟢Stable | Mapper to predict user's topic label in query. Mapper在查询中预测用户的主题标签。 | [info](operators/mapper/query_topic_detection_mapper.md) | - |
| relation_identity_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | identify relation between two entity in the text. 确定文本中两个实体之间的关系。 | - | - |
| remove_bibliography_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove bibliography at the end of documents in Latex samples. 映射器删除Latex样本中文档末尾的参考书目。 | [info](operators/mapper/remove_bibliography_mapper.md) | - |
| remove_comments_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove comments in different kinds of documents. 映射器删除不同类型的文档中的注释。 | [info](operators/mapper/remove_comments_mapper.md) | - |
| remove_header_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove headers at the beginning of documents in Latex samples. 映射器删除Latex示例中文档开头的标题。 | [info](operators/mapper/remove_header_mapper.md) | - |
| remove_long_words_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove long words within a specific range. 映射器删除特定范围内的长词。 | [info](operators/mapper/remove_long_words_mapper.md) | - |
| remove_non_chinese_character_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove non chinese Character in text samples. 映射器删除文本样本中的非中文字符。 | [info](operators/mapper/remove_non_chinese_character_mapper.md) | - |
| remove_repeat_sentences_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove repeat sentences in text samples. 映射器删除文本样本中的重复句子。 | [info](operators/mapper/remove_repeat_sentences_mapper.md) | - |
| remove_specific_chars_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean specific chars in text samples. 映射器来清理文本样本中的特定字符。 | [info](operators/mapper/remove_specific_chars_mapper.md) | - |
| remove_table_text_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove table texts from text samples. 映射器从文本样本中删除表文本。 | [info](operators/mapper/remove_table_text_mapper.md) | - |
| remove_words_with_incorrect_substrings_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove words with incorrect substrings. 映射器删除不正确的子字符串的单词。 | [info](operators/mapper/remove_words_with_incorrect_substrings_mapper.md) | - |
| replace_content_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to replace all content in the text that matches a specific regular expression pattern with a designated replacement string. 映射程序将文本中与特定正则表达式模式匹配的所有内容替换为指定的替换字符串。 | [info](operators/mapper/replace_content_mapper.md) | - |
| sdxl_prompt2prompt_mapper | 🔤Text 🚀GPU 🟢Stable | Generate pairs of similar images by the SDXL model. 通过SDXL模型生成相似图像对。 | - | - |
| sentence_augmentation_mapper | 🔤Text 🚀GPU 🧩HF 🟢Stable | Mapper to augment sentences. 映射器来增加句子。 | [info](operators/mapper/sentence_augmentation_mapper.md) | - |
| sentence_split_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to split text samples to sentences. 映射器将文本样本拆分为句子。 | [info](operators/mapper/sentence_split_mapper.md) | - |
| text_chunk_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Split input text to chunks. 将输入文本拆分为块。 | - | - |
| video_captioning_from_audio_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to caption a video according to its audio streams based on Qwen-Audio model. 映射器根据基于qwen-audio模型的音频流为视频添加字幕。 | - | - |
| video_captioning_from_frames_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to generate samples whose captions are generated based on an image-to-text model and sampled video frames. 映射器生成样本，其字幕是基于图像到文本模型和采样的视频帧生成的。 | - | - |
| video_captioning_from_summarizer_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to generate video captions by summarizing several kinds of generated texts (captions from video/audio/frames, tags from audio/frames, ...). 映射器通过总结几种生成的文本 (来自视频/音频/帧的字幕，来自音频/帧的标签，...) 来生成视频字幕。 | - | - |
| video_captioning_from_video_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to generate samples whose captions are generated based on a video-to-text model and sampled video frame. 映射器生成样本，其字幕是基于视频到文本模型和采样的视频帧生成的。 | - | - |
| video_extract_frames_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to extract frames from video files according to specified methods. 映射器根据指定的方法从视频文件中提取帧。 | - | - |
| video_face_blur_mapper | 🎬Video 💻CPU 🟢Stable | Mapper to blur faces detected in videos. 映射器模糊在视频中检测到的人脸。 | - | - |
| video_ffmpeg_wrapped_mapper | 🎬Video 💻CPU 🟢Stable | Simple wrapper for FFmpeg video filters. FFmpeg视频过滤器的简单包装。 | [info](operators/mapper/video_ffmpeg_wrapped_mapper.md) | - |
| video_remove_watermark_mapper | 🎬Video 💻CPU 🟢Stable | Remove the watermarks in videos given regions. 删除视频给定区域中的水印。 | - | - |
| video_resize_aspect_ratio_mapper | 🎬Video 💻CPU 🟢Stable | Mapper to resize videos by aspect ratio. 映射器按纵横比调整视频大小。 | [info](operators/mapper/video_resize_aspect_ratio_mapper.md) | - |
| video_resize_resolution_mapper | 🎬Video 💻CPU 🟢Stable | Mapper to resize videos resolution. 映射器来调整视频分辨率。 | [info](operators/mapper/video_resize_resolution_mapper.md) | - |
| video_split_by_duration_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to split video by duration. 映射器按持续时间分割视频。 | [info](operators/mapper/video_split_by_duration_mapper.md) | - |
| video_split_by_key_frame_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to split video by key frame. 映射器按关键帧分割视频。 | [info](operators/mapper/video_split_by_key_frame_mapper.md) | - |
| video_split_by_scene_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to cut videos into scene clips. 映射器将视频剪切成场景剪辑。 | [info](operators/mapper/video_split_by_scene_mapper.md) | - |
| video_tagging_from_audio_mapper | 🎬Video 🚀GPU 🧩HF 🟢Stable | Mapper to generate video tags from audio streams extracted by video using the Audio Spectrogram Transformer. 映射器使用音频频谱图转换器从视频提取的音频流生成视频标签。 | [info](operators/mapper/video_tagging_from_audio_mapper.md) | - |
| video_tagging_from_frames_mapper | 🎬Video 🚀GPU 🟢Stable | Mapper to generate video tags from frames extract by video. 映射器从视频提取的帧生成视频标签。 | - | - |
| whitespace_normalization_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to normalize different kinds of whitespaces to whitespace ' ' (0x20) in text samples. 映射器，用于将文本示例中的不同类型的空白标准化为空白 “” (0 x20)。 | [info](operators/mapper/whitespace_normalization_mapper.md) | - |

## selector <a name="selector"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| frequency_specified_field_selector | 💻CPU 🟢Stable | Selector to select samples based on the sorted frequency of specified field. 选择器根据指定字段的排序频率选择样本。 | [info](operators/selector/frequency_specified_field_selector.md) | - |
| random_selector | 💻CPU 🟢Stable | Selector to random select samples. 选择器来随机选择样本。 | - | - |
| range_specified_field_selector | 💻CPU 🟢Stable | Selector to select a range of samples based on the sorted specified field value from smallest to largest. 选择器根据从最小到最大的排序指定字段值选择样本范围。 | [info](operators/selector/range_specified_field_selector.md) | - |
| tags_specified_field_selector | 💻CPU 🟢Stable | Selector to select samples based on the tags of specified field. 选择器根据指定字段的标签选择样本。 | [info](operators/selector/tags_specified_field_selector.md) | - |
| topk_specified_field_selector | 💻CPU 🟢Stable | Selector to select top samples based on the sorted specified field value. 选择器根据已排序的指定字段值选择顶部样本。 | [info](operators/selector/topk_specified_field_selector.md) | - |


## Contributing  贡献

We welcome contributions of adding new operators. Please refer to [How-to Guide
for Developers](DeveloperGuide.md).

我们欢迎社区贡献新的算子，具体请参考[开发者指南](DeveloperGuide_ZH.md)。
