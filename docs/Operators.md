
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
| [filter](#filter) | 49 | Filters out low-quality samples. 过滤低质量样本。 |
| [formatter](#formatter) | 8 | Discovers, loads, and canonicalizes source data. 发现、加载、规范化原始数据。 |
| [grouper](#grouper) | 3 | Group samples to batched samples. 将样本分组，每一组组成一个批量样本。 |
| [mapper](#mapper) | 82 | Edits and transforms samples. 对数据样本进行编辑和转换。 |
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

| Operator 算子 | Tags 标签 | Description 描述 | Source code 源码 | Unit tests 单测样例 |
|----------|------|-------------|-------------|------------|
| entity_attribute_aggregator | 💻CPU 🔗API 🟢Stable | Return conclusion of the given entity's attribute from some docs. 从一些文档返回给定实体的属性的结论。 | [code](../data_juicer/ops/aggregator/entity_attribute_aggregator.py) | [tests](../tests/ops/aggregator/test_entity_attribute_aggregator.py) |
| meta_tags_aggregator | 💻CPU 🔗API 🟢Stable | Merge similar meta tags to one tag. 将类似的元标记合并到一个标记。 | [code](../data_juicer/ops/aggregator/meta_tags_aggregator.py) | [tests](../tests/ops/aggregator/test_meta_tags_aggregator.py) |
| most_relevant_entities_aggregator | 💻CPU 🔗API 🟢Stable | Extract entities closely related to a given entity from some texts, and sort them in descending order of importance. 从一些文本中提取与给定实体密切相关的实体，并按重要性的降序对它们进行排序。 | [code](../data_juicer/ops/aggregator/most_relevant_entities_aggregator.py) | [tests](../tests/ops/aggregator/test_most_relevant_entities_aggregator.py) |
| nested_aggregator | 🔤Text 💻CPU 🔗API 🟢Stable | Considering the limitation of input length, nested aggregate contents for each given number of samples. 考虑到输入长度的限制，嵌套聚合每个给定数量的样本的内容。 | [code](../data_juicer/ops/aggregator/nested_aggregator.py) | [tests](../tests/ops/aggregator/test_nested_aggregator.py) |

## deduplicator <a name="deduplicator"/>

| Operator 算子 | Tags 标签 | Description 描述 | Source code 源码 | Unit tests 单测样例 |
|----------|------|-------------|-------------|------------|
| document_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using exact matching. Deduplicator使用精确匹配在文档级别删除重复的样本。 | [code](../data_juicer/ops/deduplicator/document_deduplicator.py) | [tests](../tests/ops/deduplicator/test_document_deduplicator.py) |
| document_minhash_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using MinHashLSH. Deduplicator使用MinHashLSH在文档级别删除重复的样本。 | [code](../data_juicer/ops/deduplicator/document_minhash_deduplicator.py) | [tests](../tests/ops/deduplicator/test_document_minhash_deduplicator.py) |
| document_simhash_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using SimHash. Deduplicator使用SimHash在文档级别对样本进行重复数据删除。 | [code](../data_juicer/ops/deduplicator/document_simhash_deduplicator.py) | [tests](../tests/ops/deduplicator/test_document_simhash_deduplicator.py) |
| image_deduplicator | 🏞Image 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using exact matching of images between documents. Deduplicator使用文档之间的图像精确匹配在文档级别删除重复的样本。 | [code](../data_juicer/ops/deduplicator/image_deduplicator.py) | [tests](../tests/ops/deduplicator/test_image_deduplicator.py) |
| ray_basic_deduplicator | 💻CPU 🔴Alpha | Backend for deduplicator. deduplicator的后端。 | [code](../data_juicer/ops/deduplicator/ray_basic_deduplicator.py) | - |
| ray_bts_minhash_deduplicator | 🔤Text 💻CPU 🟡Beta | A distributed implementation of Union-Find with load balancing. 具有负载平衡的Union-Find的分布式实现。 | [code](../data_juicer/ops/deduplicator/ray_bts_minhash_deduplicator.py) | [tests](../tests/ops/deduplicator/test_ray_bts_minhash_deduplicator.py) |
| ray_document_deduplicator | 🔤Text 💻CPU 🟡Beta | Deduplicator to deduplicate samples at document-level using exact matching. Deduplicator使用精确匹配在文档级别删除重复的样本。 | [code](../data_juicer/ops/deduplicator/ray_document_deduplicator.py) | [tests](../tests/ops/deduplicator/test_ray_document_deduplicator.py) |
| ray_image_deduplicator | 🏞Image 💻CPU 🟡Beta | Deduplicator to deduplicate samples at document-level using exact matching of images between documents. Deduplicator使用文档之间的图像精确匹配在文档级别删除重复的样本。 | [code](../data_juicer/ops/deduplicator/ray_image_deduplicator.py) | [tests](../tests/ops/deduplicator/test_ray_image_deduplicator.py) |
| ray_video_deduplicator | 🎬Video 💻CPU 🟡Beta | Deduplicator to deduplicate samples at document-level using exact matching of videos between documents. Deduplicator使用文档之间的视频精确匹配在文档级别删除重复的样本。 | [code](../data_juicer/ops/deduplicator/ray_video_deduplicator.py) | [tests](../tests/ops/deduplicator/test_ray_video_deduplicator.py) |
| video_deduplicator | 🎬Video 💻CPU 🟢Stable | Deduplicator to deduplicate samples at document-level using exact matching of videos between documents. Deduplicator使用文档之间的视频精确匹配在文档级别删除重复的样本。 | [code](../data_juicer/ops/deduplicator/video_deduplicator.py) | [tests](../tests/ops/deduplicator/test_video_deduplicator.py) |

## filter <a name="filter"/>

| Operator 算子 | Tags 标签 | Description 描述 | Source code 源码 | Unit tests 单测样例 |
|----------|------|-------------|-------------|------------|
| alphanumeric_filter | 🔤Text 💻CPU 🧩HF 🟢Stable | Filter to keep samples with alphabet/numeric ratio within a specific range. 过滤器保持样品与字母/数字的比例在一个特定的范围内。 | [code](../data_juicer/ops/filter/alphanumeric_filter.py) | [tests](../tests/ops/filter/test_alphanumeric_filter.py) |
| audio_duration_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples whose audios' durations are within a specified range. 保留音频持续时间在指定范围内的数据样本。 | [code](../data_juicer/ops/filter/audio_duration_filter.py) | [tests](../tests/ops/filter/test_audio_duration_filter.py) |
| audio_nmf_snr_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples whose audios' SNRs (computed based on NMF) are within a specified range. 保留音频的snr (根据NMF计算) 在指定范围内的数据样本。 | [code](../data_juicer/ops/filter/audio_nmf_snr_filter.py) | [tests](../tests/ops/filter/test_audio_nmf_snr_filter.py) |
| audio_size_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples whose audio size (in bytes/kb/MB/...) within a specific range. 保留音频大小 (以字节/kb/MB/... 为单位) 在特定范围内的数据样本。 | [code](../data_juicer/ops/filter/audio_size_filter.py) | [tests](../tests/ops/filter/test_audio_size_filter.py) |
| average_line_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with average line length within a specific range. 过滤器，以保持平均线长度在特定范围内的样本。 | [code](../data_juicer/ops/filter/average_line_length_filter.py) | [tests](../tests/ops/filter/test_average_line_length_filter.py) |
| character_repetition_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with char-level n-gram repetition ratio within a specific range. 过滤器将具有char级n-gram重复比率的样本保持在特定范围内。 | [code](../data_juicer/ops/filter/character_repetition_filter.py) | [tests](../tests/ops/filter/test_character_repetition_filter.py) |
| flagged_words_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with flagged-word ratio less than a specific max value. 过滤以保持标记词比率小于特定最大值的样本。 | [code](../data_juicer/ops/filter/flagged_words_filter.py) | [tests](../tests/ops/filter/test_flagged_words_filter.py) |
| general_field_filter | 💻CPU 🟡Beta | Filter to keep samples based on a general field filter condition. 根据常规字段筛选条件保留样本。 | [code](../data_juicer/ops/filter/general_field_filter.py) | [tests](../tests/ops/filter/test_general_field_filter.py) |
| image_aesthetics_filter | 🏞Image 💻CPU 🧩HF 🟢Stable | Filter to keep samples with aesthetics scores within a specific range. 过滤以保持美学分数在特定范围内的样品。 | [code](../data_juicer/ops/filter/image_aesthetics_filter.py) | [tests](../tests/ops/filter/test_image_aesthetics_filter.py) |
| image_aspect_ratio_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with image aspect ratio within a specific range. 过滤器，以保持特定范围内的图像长宽比的样本。 | [code](../data_juicer/ops/filter/image_aspect_ratio_filter.py) | [tests](../tests/ops/filter/test_image_aspect_ratio_filter.py) |
| image_face_count_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with the number of faces within a specific range. 过滤以保持样本的面数在特定范围内。 | [code](../data_juicer/ops/filter/image_face_count_filter.py) | [tests](../tests/ops/filter/test_image_face_count_filter.py) |
| image_face_ratio_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with face area ratios within a specific range. 过滤以保持面面积比在特定范围内的样本。 | [code](../data_juicer/ops/filter/image_face_ratio_filter.py) | [tests](../tests/ops/filter/test_image_face_ratio_filter.py) |
| image_nsfw_filter | 🏞Image 💻CPU 🧩HF 🟢Stable | Filter to keep samples whose images have low nsfw scores. 过滤器保留图像具有低nsfw分数的样本。 | [code](../data_juicer/ops/filter/image_nsfw_filter.py) | [tests](../tests/ops/filter/test_image_nsfw_filter.py) |
| image_pair_similarity_filter | 🏞Image 💻CPU 🧩HF 🟢Stable | Filter to keep image pairs with similarities between images within a specific range. 过滤器将图像之间具有相似性的图像对保持在特定范围内。 | [code](../data_juicer/ops/filter/image_pair_similarity_filter.py) | [tests](../tests/ops/filter/test_image_pair_similarity_filter.py) |
| image_shape_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with image shape (w, h) within specific ranges. 过滤器保持样品的图像形状 (w，h) 在特定范围内。 | [code](../data_juicer/ops/filter/image_shape_filter.py) | [tests](../tests/ops/filter/test_image_shape_filter.py) |
| image_size_filter | 🏞Image 💻CPU 🟢Stable | Keep data samples whose image size (in Bytes/KB/MB/...) within a specific range. 保留图像大小 (以字节/KB/MB/... 为单位) 在特定范围内的数据样本。 | [code](../data_juicer/ops/filter/image_size_filter.py) | [tests](../tests/ops/filter/test_image_size_filter.py) |
| image_text_matching_filter | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Filter to keep samples those matching score between image and text within a specific range. 过滤器将图像和文本之间的匹配分数保持在特定范围内。 | [code](../data_juicer/ops/filter/image_text_matching_filter.py) | [tests](../tests/ops/filter/test_image_text_matching_filter.py) |
| image_text_similarity_filter | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Filter to keep samples those similarities between image and text within a specific range. 过滤器将图像和文本之间的相似性保持在特定范围内。 | [code](../data_juicer/ops/filter/image_text_similarity_filter.py) | [tests](../tests/ops/filter/test_image_text_similarity_filter.py) |
| image_watermark_filter | 🏞Image 💻CPU 🧩HF 🟢Stable | Filter to keep samples whose images have no watermark with high probability. 过滤器，以保留图像没有水印的样本。 | [code](../data_juicer/ops/filter/image_watermark_filter.py) | [tests](../tests/ops/filter/test_image_watermark_filter.py) |
| language_id_score_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples in a specific language with confidence score larger than a specific min value. 过滤器以保留置信度得分大于特定最小值的特定语言的样本。 | [code](../data_juicer/ops/filter/language_id_score_filter.py) | [tests](../tests/ops/filter/test_language_id_score_filter.py) |
| llm_analysis_filter | 💻CPU 🌊vLLM 🧩HF 🔗API 🟡Beta | Base filter class for leveraging LLMs to filter various samples. 用于利用llm过滤各种样本的基本筛选器类。 | [code](../data_juicer/ops/filter/llm_analysis_filter.py) | [tests](../tests/ops/filter/test_llm_analysis_filter.py) |
| llm_difficulty_score_filter | 💻CPU 🟡Beta | Filter to keep sample with high difficulty score estimated by LLM. 过滤器以保持LLM估计的高难度分数的样本。 | [code](../data_juicer/ops/filter/llm_difficulty_score_filter.py) | [tests](../tests/ops/filter/test_llm_difficulty_score_filter.py) |
| llm_quality_score_filter | 💻CPU 🟡Beta | Filter to keep sample with high quality score estimated by LLM. 过滤器以保持LLM估计的高质量分数的样本。 | [code](../data_juicer/ops/filter/llm_quality_score_filter.py) | [tests](../tests/ops/filter/test_llm_quality_score_filter.py) |
| maximum_line_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with maximum line length within a specific range. 过滤器将最大行长度的样本保持在特定范围内。 | [code](../data_juicer/ops/filter/maximum_line_length_filter.py) | [tests](../tests/ops/filter/test_maximum_line_length_filter.py) |
| perplexity_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with perplexity score less than a specific max value. 过滤以保留困惑度分数小于特定最大值的样本。 | [code](../data_juicer/ops/filter/perplexity_filter.py) | [tests](../tests/ops/filter/test_perplexity_filter.py) |
| phrase_grounding_recall_filter | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Filter to keep samples whose locating recalls of phrases extracted from text in the images are within a specified range. 过滤器，用于保留从图像中的文本中提取的短语的定位回忆在指定范围内的样本。 | [code](../data_juicer/ops/filter/phrase_grounding_recall_filter.py) | [tests](../tests/ops/filter/test_phrase_grounding_recall_filter.py) |
| special_characters_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with special-char ratio within a specific range. 过滤器将具有特殊字符比率的样品保持在特定范围内。 | [code](../data_juicer/ops/filter/special_characters_filter.py) | [tests](../tests/ops/filter/test_special_characters_filter.py) |
| specified_field_filter | 💻CPU 🟢Stable | Filter based on specified field information. 根据指定的字段信息进行筛选。 | [code](../data_juicer/ops/filter/specified_field_filter.py) | [tests](../tests/ops/filter/test_specified_field_filter.py) |
| specified_numeric_field_filter | 💻CPU 🟢Stable | Filter based on specified numeric field information. 根据指定的数值字段信息进行筛选。 | [code](../data_juicer/ops/filter/specified_numeric_field_filter.py) | [tests](../tests/ops/filter/test_specified_numeric_field_filter.py) |
| stopwords_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with stopword ratio larger than a specific min value. 过滤以保持停止词比率大于特定最小值的样本。 | [code](../data_juicer/ops/filter/stopwords_filter.py) | [tests](../tests/ops/filter/test_stopwords_filter.py) |
| suffix_filter | 💻CPU 🟢Stable | Filter to keep samples with specified suffix. 过滤器以保留具有指定后缀的样本。 | [code](../data_juicer/ops/filter/suffix_filter.py) | [tests](../tests/ops/filter/test_suffix_filter.py) |
| text_action_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep texts those contain actions in the text. 过滤以保留文本中包含操作的文本。 | [code](../data_juicer/ops/filter/text_action_filter.py) | [tests](../tests/ops/filter/test_text_action_filter.py) |
| text_entity_dependency_filter | 🔤Text 💻CPU 🟢Stable | Identify the entities in the text which are independent with other token, and filter them. 识别文本中与其他令牌独立的实体，并对其进行过滤。 | [code](../data_juicer/ops/filter/text_entity_dependency_filter.py) | [tests](../tests/ops/filter/test_text_entity_dependency_filter.py) |
| text_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with total text length within a specific range. 过滤以保持文本总长度在特定范围内的样本。 | [code](../data_juicer/ops/filter/text_length_filter.py) | [tests](../tests/ops/filter/test_text_length_filter.py) |
| text_pair_similarity_filter | 🔤Text 💻CPU 🧩HF 🟢Stable | Filter to keep text pairs with similarities between texts within a specific range. 过滤器将文本之间具有相似性的文本对保留在特定范围内。 | [code](../data_juicer/ops/filter/text_pair_similarity_filter.py) | [tests](../tests/ops/filter/test_text_pair_similarity_filter.py) |
| token_num_filter | 🔤Text 💻CPU 🧩HF 🟢Stable | Filter to keep samples with total token number within a specific range. 筛选器将总令牌数的样本保留在特定范围内。 | [code](../data_juicer/ops/filter/token_num_filter.py) | [tests](../tests/ops/filter/test_token_num_filter.py) |
| video_aesthetics_filter | 🎬Video 💻CPU 🧩HF 🟢Stable | Filter to keep data samples with aesthetics scores for specified frames in the videos within a specific range. 过滤器将视频中指定帧的美学得分数据样本保留在特定范围内。 | [code](../data_juicer/ops/filter/video_aesthetics_filter.py) | [tests](../tests/ops/filter/test_video_aesthetics_filter.py) |
| video_aspect_ratio_filter | 🎬Video 💻CPU 🟢Stable | Filter to keep samples with video aspect ratio within a specific range. 过滤器将视频纵横比的样本保持在特定范围内。 | [code](../data_juicer/ops/filter/video_aspect_ratio_filter.py) | [tests](../tests/ops/filter/test_video_aspect_ratio_filter.py) |
| video_duration_filter | 🎬Video 💻CPU 🟢Stable | Keep data samples whose videos' durations are within a specified range. 保留视频持续时间在指定范围内的数据样本。 | [code](../data_juicer/ops/filter/video_duration_filter.py) | [tests](../tests/ops/filter/test_video_duration_filter.py) |
| video_frames_text_similarity_filter | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Filter to keep samples those similarities between sampled video frame images and text within a specific range. 过滤以保持采样视频帧图像和文本之间的相似性在特定范围内。 | [code](../data_juicer/ops/filter/video_frames_text_similarity_filter.py) | [tests](../tests/ops/filter/test_video_frames_text_similarity_filter.py) |
| video_motion_score_filter | 🎬Video 💻CPU 🟢Stable | Filter to keep samples with video motion scores within a specific range. 过滤器将视频运动分数的样本保持在特定范围内。 | [code](../data_juicer/ops/filter/video_motion_score_filter.py) | [tests](../tests/ops/filter/test_video_motion_score_filter.py) |
| video_motion_score_raft_filter | 🎬Video 💻CPU 🟢Stable | Filter to keep samples with video motion scores within a specified range. 过滤器将视频运动分数的样本保持在指定范围内。 | [code](../data_juicer/ops/filter/video_motion_score_raft_filter.py) | [tests](../tests/ops/filter/test_video_motion_score_raft_filter.py) |
| video_nsfw_filter | 🎬Video 💻CPU 🧩HF 🟢Stable | Filter to keep samples whose videos have low nsfw scores. 过滤器以保留其视频具有低nsfw分数的样本。 | [code](../data_juicer/ops/filter/video_nsfw_filter.py) | [tests](../tests/ops/filter/test_video_nsfw_filter.py) |
| video_ocr_area_ratio_filter | 🎬Video 💻CPU 🟢Stable | Keep data samples whose detected text area ratios for specified frames in the video are within a specified range. 保留检测到的视频中指定帧的文本面积比率在指定范围内的数据样本。 | [code](../data_juicer/ops/filter/video_ocr_area_ratio_filter.py) | [tests](../tests/ops/filter/test_video_ocr_area_ratio_filter.py) |
| video_resolution_filter | 🎬Video 💻CPU 🟢Stable | Keep data samples whose videos' resolutions are within a specified range. 保留视频分辨率在指定范围内的数据样本。 | [code](../data_juicer/ops/filter/video_resolution_filter.py) | [tests](../tests/ops/filter/test_video_resolution_filter.py) |
| video_tagging_from_frames_filter | 🎬Video 💻CPU 🟢Stable | Filter to keep samples whose videos contain the given tags. 过滤器以保留其视频包含给定标签的样本。 | [code](../data_juicer/ops/filter/video_tagging_from_frames_filter.py) | [tests](../tests/ops/filter/test_video_tagging_from_frames_filter.py) |
| video_watermark_filter | 🎬Video 💻CPU 🧩HF 🟢Stable | Filter to keep samples whose videos have no watermark with high probability. 过滤器以保持其视频具有高概率没有水印的样本。 | [code](../data_juicer/ops/filter/video_watermark_filter.py) | [tests](../tests/ops/filter/test_video_watermark_filter.py) |
| word_repetition_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with word-level n-gram repetition ratio within a specific range. 过滤器将单词级n-gram重复比率的样本保持在特定范围内。 | [code](../data_juicer/ops/filter/word_repetition_filter.py) | [tests](../tests/ops/filter/test_word_repetition_filter.py) |
| words_num_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with total words number within a specific range. 过滤器，以保持总字数在特定范围内的样本。 | [code](../data_juicer/ops/filter/words_num_filter.py) | [tests](../tests/ops/filter/test_words_num_filter.py) |

## formatter <a name="formatter"/>

| Operator 算子 | Tags 标签 | Description 描述 | Source code 源码 | Unit tests 单测样例 |
|----------|------|-------------|-------------|------------|
| csv_formatter | 🟢Stable | The class is used to load and format csv-type files. 类用于加载和格式化csv类型的文件。 | [code](../data_juicer/format/csv_formatter.py) | [tests](../tests/format/test_csv_formatter.py) |
| empty_formatter | 🟢Stable | The class is used to create empty data. 类用于创建空数据。 | [code](../data_juicer/format/empty_formatter.py) | [tests](../tests/format/test_empty_formatter.py) |
| json_formatter | 🟡Beta | The class is used to load and format json-type files. 类用于加载和格式化json类型的文件。 | [code](../data_juicer/format/json_formatter.py) | [tests](../tests/format/test_json_formatter.py) |
| local_formatter | 🟢Stable | The class is used to load a dataset from local files or local directory. 类用于从本地文件或本地目录加载数据集。 | [code](../data_juicer/format/formatter.py) | [tests](../tests/format/test_unify_format.py) |
| parquet_formatter | 🟢Stable | The class is used to load and format parquet-type files. 该类用于加载和格式化镶木地板类型的文件。 | [code](../data_juicer/format/parquet_formatter.py) | [tests](../tests/format/test_parquet_formatter.py) |
| remote_formatter | 🟢Stable | The class is used to load a dataset from repository of huggingface hub. 该类用于从huggingface hub的存储库加载数据集。 | [code](../data_juicer/format/formatter.py) | [tests](../tests/format/test_unify_format.py) |
| text_formatter | 🔴Alpha | The class is used to load and format text-type files. 类用于加载和格式化文本类型文件。 | [code](../data_juicer/format/text_formatter.py) | - |
| tsv_formatter | 🟢Stable | The class is used to load and format tsv-type files. 该类用于加载和格式化tsv类型的文件。 | [code](../data_juicer/format/tsv_formatter.py) | [tests](../tests/format/test_tsv_formatter.py) |

## grouper <a name="grouper"/>

| Operator 算子 | Tags 标签 | Description 描述 | Source code 源码 | Unit tests 单测样例 |
|----------|------|-------------|-------------|------------|
| key_value_grouper | 🔤Text 💻CPU 🟢Stable | Group samples to batched samples according values in given keys. 根据给定键中的值将样本分组为批处理样本。 | [code](../data_juicer/ops/grouper/key_value_grouper.py) | [tests](../tests/ops/grouper/test_key_value_grouper.py) |
| naive_grouper | 💻CPU 🟢Stable | Group all samples to one batched sample. 将所有样品分组为一批样品。 | [code](../data_juicer/ops/grouper/naive_grouper.py) | [tests](../tests/ops/grouper/test_naive_grouper.py) |
| naive_reverse_grouper | 💻CPU 🟢Stable | Split batched samples to samples. 将批处理的样品拆分为样品。 | [code](../data_juicer/ops/grouper/naive_reverse_grouper.py) | [tests](../tests/ops/grouper/test_naive_reverse_grouper.py) |

## mapper <a name="mapper"/>

| Operator 算子 | Tags 标签 | Description 描述 | Source code 源码 | Unit tests 单测样例 |
|----------|------|-------------|-------------|------------|
| audio_add_gaussian_noise_mapper | 📣Audio 💻CPU 🟡Beta | Mapper to add gaussian noise to audio. 映射器向音频添加高斯噪声。 | [code](../data_juicer/ops/mapper/audio_add_gaussian_noise_mapper.py) | [tests](../tests/ops/mapper/test_audio_add_gaussian_noise_mapper.py) |
| audio_ffmpeg_wrapped_mapper | 📣Audio 💻CPU 🟢Stable | Simple wrapper for FFmpeg audio filters. FFmpeg音频滤波器的简单包装。 | [code](../data_juicer/ops/mapper/audio_ffmpeg_wrapped_mapper.py) | [tests](../tests/ops/mapper/test_audio_ffmpeg_wrapped_mapper.py) |
| calibrate_qa_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Mapper to calibrate question-answer pairs based on reference text. 映射器基于参考文本校准问题-答案对。 | [code](../data_juicer/ops/mapper/calibrate_qa_mapper.py) | [tests](../tests/ops/mapper/test_calibrate_qa_mapper.py) |
| calibrate_query_mapper | 💻CPU 🟢Stable | Mapper to calibrate query in question-answer pairs based on reference text. 映射器基于参考文本校准问答对中的查询。 | [code](../data_juicer/ops/mapper/calibrate_query_mapper.py) | [tests](../tests/ops/mapper/test_calibrate_query_mapper.py) |
| calibrate_response_mapper | 💻CPU 🟢Stable | Mapper to calibrate response in question-answer pairs based on reference text. 映射器基于参考文本校准问答对中的响应。 | [code](../data_juicer/ops/mapper/calibrate_response_mapper.py) | [tests](../tests/ops/mapper/test_calibrate_response_mapper.py) |
| chinese_convert_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to convert Chinese between Traditional Chinese, Simplified Chinese and Japanese Kanji. 映射器在繁体中文，简体中文和日语汉字之间转换中文。 | [code](../data_juicer/ops/mapper/chinese_convert_mapper.py) | [tests](../tests/ops/mapper/test_chinese_convert_mapper.py) |
| clean_copyright_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean copyright comments at the beginning of the text samples. Mapper清理版权注释开头的文本样本。 | [code](../data_juicer/ops/mapper/clean_copyright_mapper.py) | [tests](../tests/ops/mapper/test_clean_copyright_mapper.py) |
| clean_email_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean email in text samples. 映射器清理文本样本中的电子邮件。 | [code](../data_juicer/ops/mapper/clean_email_mapper.py) | [tests](../tests/ops/mapper/test_clean_email_mapper.py) |
| clean_html_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean html code in text samples. 映射器来清理文本示例中的html代码。 | [code](../data_juicer/ops/mapper/clean_html_mapper.py) | [tests](../tests/ops/mapper/test_clean_html_mapper.py) |
| clean_ip_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean ipv4 and ipv6 address in text samples. 映射器以清除文本示例中的ipv4和ipv6地址。 | [code](../data_juicer/ops/mapper/clean_ip_mapper.py) | [tests](../tests/ops/mapper/test_clean_ip_mapper.py) |
| clean_links_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean links like http/https/ftp in text samples. 映射器来清理链接，如文本示例中的http/https/ftp。 | [code](../data_juicer/ops/mapper/clean_links_mapper.py) | [tests](../tests/ops/mapper/test_clean_links_mapper.py) |
| dialog_intent_detection_mapper | 💻CPU 🔗API 🟢Stable | Mapper to generate user's intent labels in dialog. 映射器在对话框中生成用户的意图标签。 | [code](../data_juicer/ops/mapper/dialog_intent_detection_mapper.py) | [tests](../tests/ops/mapper/test_dialog_intent_detection_mapper.py) |
| dialog_sentiment_detection_mapper | 💻CPU 🔗API 🟢Stable | Mapper to generate user's sentiment labels in dialog. 映射器在对话框中生成用户的情绪标签。 | [code](../data_juicer/ops/mapper/dialog_sentiment_detection_mapper.py) | [tests](../tests/ops/mapper/test_dialog_sentiment_detection_mapper.py) |
| dialog_sentiment_intensity_mapper | 💻CPU 🔗API 🟢Stable | Mapper to predict user's sentiment intensity (from -5 to 5 in default prompt) in dialog. Mapper在对话框中预测用户的情绪强度 (在默认提示中从-5到5)。 | [code](../data_juicer/ops/mapper/dialog_sentiment_intensity_mapper.py) | [tests](../tests/ops/mapper/test_dialog_sentiment_intensity_mapper.py) |
| dialog_topic_detection_mapper | 💻CPU 🔗API 🟢Stable | Mapper to generate user's topic labels in dialog. 映射器在对话框中生成用户的主题标签。 | [code](../data_juicer/ops/mapper/dialog_topic_detection_mapper.py) | [tests](../tests/ops/mapper/test_dialog_topic_detection_mapper.py) |
| download_file_mapper | 💻CPU 🟡Beta | Mapper to download url files to local files or load them into memory. 映射器将url文件下载到本地文件或将其加载到内存中。 | [code](../data_juicer/ops/mapper/download_file_mapper.py) | [tests](../tests/ops/mapper/test_download_file_mapper.py) |
| expand_macro_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to expand macro definitions in the document body of Latex samples. Mapper来扩展Latex示例文档主体中的宏定义。 | [code](../data_juicer/ops/mapper/expand_macro_mapper.py) | [tests](../tests/ops/mapper/test_expand_macro_mapper.py) |
| extract_entity_attribute_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract attributes for given entities from the text. 从文本中提取给定实体的属性。 | [code](../data_juicer/ops/mapper/extract_entity_attribute_mapper.py) | [tests](../tests/ops/mapper/test_extract_entity_attribute_mapper.py) |
| extract_entity_relation_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract entities and relations in the text for knowledge graph. 提取知识图谱的文本中的实体和关系。 | [code](../data_juicer/ops/mapper/extract_entity_relation_mapper.py) | [tests](../tests/ops/mapper/test_extract_entity_relation_mapper.py) |
| extract_event_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract events and relevant characters in the text. 提取文本中的事件和相关字符。 | [code](../data_juicer/ops/mapper/extract_event_mapper.py) | [tests](../tests/ops/mapper/test_extract_event_mapper.py) |
| extract_keyword_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Generate keywords for the text. 为文本生成关键字。 | [code](../data_juicer/ops/mapper/extract_keyword_mapper.py) | [tests](../tests/ops/mapper/test_extract_keyword_mapper.py) |
| extract_nickname_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract nickname relationship in the text. 提取文本中的昵称关系。 | [code](../data_juicer/ops/mapper/extract_nickname_mapper.py) | [tests](../tests/ops/mapper/test_extract_nickname_mapper.py) |
| extract_support_text_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extract support sub text for a summary. 提取摘要的支持子文本。 | [code](../data_juicer/ops/mapper/extract_support_text_mapper.py) | [tests](../tests/ops/mapper/test_extract_support_text_mapper.py) |
| extract_tables_from_html_mapper | 🔤Text 💻CPU 🟡Beta | Mapper to extract tables from HTML content. 映射器从HTML内容中提取表。 | [code](../data_juicer/ops/mapper/extract_tables_from_html_mapper.py) | [tests](../tests/ops/mapper/test_extract_tables_from_html_mapper.py) |
| fix_unicode_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to fix unicode errors in text samples. 映射器修复文本示例中的unicode错误。 | [code](../data_juicer/ops/mapper/fix_unicode_mapper.py) | [tests](../tests/ops/mapper/test_fix_unicode_mapper.py) |
| generate_qa_from_examples_mapper | 💻CPU 🌊vLLM 🧩HF 🟢Stable | Mapper to generate question and answer pairs from examples. 映射器从示例生成问题和答案对。 | [code](../data_juicer/ops/mapper/generate_qa_from_examples_mapper.py) | [tests](../tests/ops/mapper/test_generate_qa_from_examples_mapper.py) |
| generate_qa_from_text_mapper | 🔤Text 💻CPU 🌊vLLM 🧩HF 🟢Stable | Mapper to generate question and answer pairs from text. 映射器从文本生成问题和答案对。 | [code](../data_juicer/ops/mapper/generate_qa_from_text_mapper.py) | [tests](../tests/ops/mapper/test_generate_qa_from_text_mapper.py) |
| image_blur_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to blur images. 映射器来模糊图像。 | [code](../data_juicer/ops/mapper/image_blur_mapper.py) | [tests](../tests/ops/mapper/test_image_blur_mapper.py) |
| image_captioning_from_gpt4v_mapper | 🔮Multimodal 💻CPU 🟡Beta | Mapper to generate samples whose texts are generated based on gpt-4-vision and the image. Mapper生成样本，其文本基于gpt-4-vision和图像生成。 | [code](../data_juicer/ops/mapper/image_captioning_from_gpt4v_mapper.py) | [tests](../tests/ops/mapper/test_image_captioning_from_gpt4v_mapper.py) |
| image_captioning_mapper | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Mapper to generate samples whose captions are generated based on another model and the figure. 映射器生成样本，其标题是基于另一个模型和图生成的。 | [code](../data_juicer/ops/mapper/image_captioning_mapper.py) | [tests](../tests/ops/mapper/test_image_captioning_mapper.py) |
| image_diffusion_mapper | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Generate image by diffusion model. 通过扩散模型生成图像。 | [code](../data_juicer/ops/mapper/image_diffusion_mapper.py) | [tests](../tests/ops/mapper/test_image_diffusion_mapper.py) |
| image_face_blur_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to blur faces detected in images. 映射器模糊图像中检测到的人脸。 | [code](../data_juicer/ops/mapper/image_face_blur_mapper.py) | [tests](../tests/ops/mapper/test_image_face_blur_mapper.py) |
| image_remove_background_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to remove background of images. 映射器删除图像的背景。 | [code](../data_juicer/ops/mapper/image_remove_background_mapper.py) | [tests](../tests/ops/mapper/test_image_remove_background_mapper.py) |
| image_segment_mapper | 🏞Image 💻CPU 🟢Stable | Perform segment-anything on images and return the bounding boxes. 在图像上执行segment-anything并返回边界框。 | [code](../data_juicer/ops/mapper/image_segment_mapper.py) | [tests](../tests/ops/mapper/test_image_segment_mapper.py) |
| image_tagging_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to generate image tags. 映射器生成图像标签。 | [code](../data_juicer/ops/mapper/image_tagging_mapper.py) | [tests](../tests/ops/mapper/test_image_tagging_mapper.py) |
| imgdiff_difference_area_generator_mapper | 💻CPU 🟡Beta | A fused operator for OPs that is used to run sequential OPs on the same batch to allow fine-grained control on data processing. OPs的融合操作符，用于在同一批次上运行顺序OPs，以实现对数据处理的细粒度控制。 | [code](../data_juicer/ops/mapper/imgdiff_difference_area_generator_mapper.py) | [tests](../tests/ops/mapper/test_imgdiff_difference_area_generator_mapper.py) |
| imgdiff_difference_caption_generator_mapper | 💻CPU 🟡Beta | A fused operator for OPs that is used to run sequential OPs on the same batch to allow fine-grained control on data processing. OPs的融合操作符，用于在同一批次上运行顺序OPs，以实现对数据处理的细粒度控制。 | [code](../data_juicer/ops/mapper/imgdiff_difference_caption_generator_mapper.py) | [tests](../tests/ops/mapper/test_imgdiff_difference_caption_generator_mapper.py) |
| lidar_detection_mapper | 💻CPU 🔴Alpha | Mapper to detect ground truth from LiDAR data. 映射器从激光雷达数据中检测地面真相。 | [code](../data_juicer/ops/mapper/lidar_detection_mapper.py) | - |
| mllm_mapper | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Mapper to use MLLMs for visual question answering tasks. Mapper使用MLLMs进行视觉问答任务。 | [code](../data_juicer/ops/mapper/mllm_mapper.py) | [tests](../tests/ops/mapper/test_mllm_mapper.py) |
| nlpaug_en_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to simply augment samples in English based on nlpaug library. 映射器基于nlpaug库简单地增加英语样本。 | [code](../data_juicer/ops/mapper/nlpaug_en_mapper.py) | [tests](../tests/ops/mapper/test_nlpaug_en_mapper.py) |
| nlpcda_zh_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to simply augment samples in Chinese based on nlpcda library. 基于nlpcda库的映射器可以简单地增加中文样本。 | [code](../data_juicer/ops/mapper/nlpcda_zh_mapper.py) | [tests](../tests/ops/mapper/test_nlpcda_zh_mapper.py) |
| optimize_qa_mapper | 💻CPU 🌊vLLM 🧩HF 🟢Stable | Mapper to optimize question-answer pairs. 映射器来优化问题-答案对。 | [code](../data_juicer/ops/mapper/optimize_qa_mapper.py) | [tests](../tests/ops/mapper/test_optimize_qa_mapper.py) |
| optimize_query_mapper | 💻CPU 🟢Stable | Mapper to optimize query in question-answer pairs. 映射器来优化问答对中的查询。 | [code](../data_juicer/ops/mapper/optimize_query_mapper.py) | [tests](../tests/ops/mapper/test_optimize_query_mapper.py) |
| optimize_response_mapper | 💻CPU 🟢Stable | Mapper to optimize response in question-answer pairs. 映射器来优化问答对中的响应。 | [code](../data_juicer/ops/mapper/optimize_response_mapper.py) | [tests](../tests/ops/mapper/test_optimize_response_mapper.py) |
| pair_preference_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Mapper to construct paired preference samples. 映射器来构造成对的偏好样本。 | [code](../data_juicer/ops/mapper/pair_preference_mapper.py) | [tests](../tests/ops/mapper/test_pair_preference_mapper.py) |
| punctuation_normalization_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to normalize unicode punctuations to English punctuations in text samples. 映射器将文本示例中的unicode标点规范化为英文标点。 | [code](../data_juicer/ops/mapper/punctuation_normalization_mapper.py) | [tests](../tests/ops/mapper/test_punctuation_normalization_mapper.py) |
| python_file_mapper | 💻CPU 🟢Stable | Mapper for executing Python function defined in a file. Mapper用于执行文件中定义的Python函数。 | [code](../data_juicer/ops/mapper/python_file_mapper.py) | [tests](../tests/ops/mapper/test_python_file_mapper.py) |
| python_lambda_mapper | 💻CPU 🟢Stable | Mapper for executing Python lambda function on data samples. 用于对数据示例执行Python lambda函数的映射器。 | [code](../data_juicer/ops/mapper/python_lambda_mapper.py) | [tests](../tests/ops/mapper/test_python_lambda_mapper.py) |
| query_intent_detection_mapper | 💻CPU 🧩HF 🧩HF 🟢Stable | Mapper to predict user's Intent label in query. Mapper在查询中预测用户的意图标签。 | [code](../data_juicer/ops/mapper/query_intent_detection_mapper.py) | [tests](../tests/ops/mapper/test_query_intent_detection_mapper.py) |
| query_sentiment_detection_mapper | 💻CPU 🧩HF 🧩HF 🟢Stable | Mapper to predict user's sentiment label ('negative', 'neutral' and 'positive') in query. Mapper在查询中预测用户的情绪标签 (“负面”，“中性” 和 “正面”)。 | [code](../data_juicer/ops/mapper/query_sentiment_detection_mapper.py) | [tests](../tests/ops/mapper/test_query_sentiment_detection_mapper.py) |
| query_topic_detection_mapper | 💻CPU 🧩HF 🧩HF 🟢Stable | Mapper to predict user's topic label in query. Mapper在查询中预测用户的主题标签。 | [code](../data_juicer/ops/mapper/query_topic_detection_mapper.py) | [tests](../tests/ops/mapper/test_query_topic_detection_mapper.py) |
| relation_identity_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | identify relation between two entity in the text. 确定文本中两个实体之间的关系。 | [code](../data_juicer/ops/mapper/relation_identity_mapper.py) | [tests](../tests/ops/mapper/test_relation_identity_mapper.py) |
| remove_bibliography_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove bibliography at the end of documents in Latex samples. 映射器删除Latex样本中文档末尾的参考书目。 | [code](../data_juicer/ops/mapper/remove_bibliography_mapper.py) | [tests](../tests/ops/mapper/test_remove_bibliography_mapper.py) |
| remove_comments_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove comments in different kinds of documents. 映射器删除不同类型的文档中的注释。 | [code](../data_juicer/ops/mapper/remove_comments_mapper.py) | [tests](../tests/ops/mapper/test_remove_comments_mapper.py) |
| remove_header_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove headers at the beginning of documents in Latex samples. 映射器删除Latex示例中文档开头的标题。 | [code](../data_juicer/ops/mapper/remove_header_mapper.py) | [tests](../tests/ops/mapper/test_remove_header_mapper.py) |
| remove_long_words_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove long words within a specific range. 映射器删除特定范围内的长词。 | [code](../data_juicer/ops/mapper/remove_long_words_mapper.py) | [tests](../tests/ops/mapper/test_remove_long_words_mapper.py) |
| remove_non_chinese_character_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove non chinese Character in text samples. 映射器删除文本样本中的非中文字符。 | [code](../data_juicer/ops/mapper/remove_non_chinese_character_mapper.py) | [tests](../tests/ops/mapper/test_remove_non_chinese_character_mapper.py) |
| remove_repeat_sentences_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove repeat sentences in text samples. 映射器删除文本样本中的重复句子。 | [code](../data_juicer/ops/mapper/remove_repeat_sentences_mapper.py) | [tests](../tests/ops/mapper/test_remove_repeat_sentences_mapper.py) |
| remove_specific_chars_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean specific chars in text samples. 映射器来清理文本示例中的特定字符。 | [code](../data_juicer/ops/mapper/remove_specific_chars_mapper.py) | [tests](../tests/ops/mapper/test_remove_specific_chars_mapper.py) |
| remove_table_text_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove table texts from text samples. 映射器从文本样本中删除表文本。 | [code](../data_juicer/ops/mapper/remove_table_text_mapper.py) | [tests](../tests/ops/mapper/test_remove_table_text_mapper.py) |
| remove_words_with_incorrect_substrings_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove words with incorrect substrings. 映射器删除不正确的子字符串的单词。 | [code](../data_juicer/ops/mapper/remove_words_with_incorrect_substrings_mapper.py) | [tests](../tests/ops/mapper/test_remove_words_with_incorrect_substrings_mapper.py) |
| replace_content_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to replace all content in the text that matches a specific regular expression pattern with a designated replacement string. 映射程序将文本中与特定正则表达式模式匹配的所有内容替换为指定的替换字符串。 | [code](../data_juicer/ops/mapper/replace_content_mapper.py) | [tests](../tests/ops/mapper/test_replace_content_mapper.py) |
| sdxl_prompt2prompt_mapper | 🔤Text 💻CPU 🟢Stable | Generate pairs of similar images by the SDXL model. 通过SDXL模型生成相似图像对。 | [code](../data_juicer/ops/mapper/sdxl_prompt2prompt_mapper.py) | [tests](../tests/ops/mapper/test_sdxl_prompt2prompt_mapper.py) |
| sentence_augmentation_mapper | 🔤Text 💻CPU 🧩HF 🟢Stable | Mapper to augment sentences. 映射器来增加句子。 | [code](../data_juicer/ops/mapper/sentence_augmentation_mapper.py) | [tests](../tests/ops/mapper/test_sentence_augmentation_mapper.py) |
| sentence_split_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to split text samples to sentences. 映射器将文本样本拆分为句子。 | [code](../data_juicer/ops/mapper/sentence_split_mapper.py) | [tests](../tests/ops/mapper/test_sentence_split_mapper.py) |
| text_chunk_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Split input text to chunks. 将输入文本拆分为块。 | [code](../data_juicer/ops/mapper/text_chunk_mapper.py) | [tests](../tests/ops/mapper/test_text_chunk_mapper.py) |
| video_captioning_from_audio_mapper | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Mapper to caption a video according to its audio streams based on Qwen-Audio model. 映射器根据基于qwen-audio模型的音频流为视频添加字幕。 | [code](../data_juicer/ops/mapper/video_captioning_from_audio_mapper.py) | [tests](../tests/ops/mapper/test_video_captioning_from_audio_mapper.py) |
| video_captioning_from_frames_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to generate samples whose captions are generated based on an image-to-text model and sampled video frames. 映射器生成样本，其字幕是基于图像到文本模型和采样的视频帧生成的。 | [code](../data_juicer/ops/mapper/video_captioning_from_frames_mapper.py) | [tests](../tests/ops/mapper/test_video_captioning_from_frames_mapper.py) |
| video_captioning_from_summarizer_mapper | 🔮Multimodal 💻CPU 🧩HF 🟢Stable | Mapper to generate video captions by summarizing several kinds of generated texts (captions from video/audio/frames, tags from audio/frames, ...). 映射器通过总结几种生成的文本 (来自视频/音频/帧的字幕，来自音频/帧的标签，...) 来生成视频字幕。 | [code](../data_juicer/ops/mapper/video_captioning_from_summarizer_mapper.py) | [tests](../tests/ops/mapper/test_video_captioning_from_summarizer_mapper.py) |
| video_captioning_from_video_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to generate samples whose captions are generated based on a video-to-text model and sampled video frame. 映射器生成样本，其字幕是基于视频到文本模型和采样的视频帧生成的。 | [code](../data_juicer/ops/mapper/video_captioning_from_video_mapper.py) | [tests](../tests/ops/mapper/test_video_captioning_from_video_mapper.py) |
| video_extract_frames_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to extract frames from video files according to specified methods. 映射器根据指定的方法从视频文件中提取帧。 | [code](../data_juicer/ops/mapper/video_extract_frames_mapper.py) | [tests](../tests/ops/mapper/test_video_extract_frames_mapper.py) |
| video_face_blur_mapper | 🎬Video 💻CPU 🟢Stable | Mapper to blur faces detected in videos. 映射器模糊在视频中检测到的人脸。 | [code](../data_juicer/ops/mapper/video_face_blur_mapper.py) | [tests](../tests/ops/mapper/test_video_face_blur_mapper.py) |
| video_ffmpeg_wrapped_mapper | 🎬Video 💻CPU 🟢Stable | Simple wrapper for FFmpeg video filters. FFmpeg视频过滤器的简单包装。 | [code](../data_juicer/ops/mapper/video_ffmpeg_wrapped_mapper.py) | [tests](../tests/ops/mapper/test_video_ffmpeg_wrapped_mapper.py) |
| video_remove_watermark_mapper | 🎬Video 💻CPU 🟢Stable | Remove the watermarks in videos given regions. 删除视频给定区域中的水印。 | [code](../data_juicer/ops/mapper/video_remove_watermark_mapper.py) | [tests](../tests/ops/mapper/test_video_remove_watermark_mapper.py) |
| video_resize_aspect_ratio_mapper | 🎬Video 💻CPU 🟢Stable | Mapper to resize videos by aspect ratio. 映射器按纵横比调整视频大小。 | [code](../data_juicer/ops/mapper/video_resize_aspect_ratio_mapper.py) | [tests](../tests/ops/mapper/test_video_resize_aspect_ratio_mapper.py) |
| video_resize_resolution_mapper | 🎬Video 💻CPU 🟢Stable | Mapper to resize videos resolution. 映射器来调整视频分辨率。 | [code](../data_juicer/ops/mapper/video_resize_resolution_mapper.py) | [tests](../tests/ops/mapper/test_video_resize_resolution_mapper.py) |
| video_split_by_duration_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to split video by duration. 映射器按持续时间分割视频。 | [code](../data_juicer/ops/mapper/video_split_by_duration_mapper.py) | [tests](../tests/ops/mapper/test_video_split_by_duration_mapper.py) |
| video_split_by_key_frame_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to split video by key frame. 映射器按关键帧分割视频。 | [code](../data_juicer/ops/mapper/video_split_by_key_frame_mapper.py) | [tests](../tests/ops/mapper/test_video_split_by_key_frame_mapper.py) |
| video_split_by_scene_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to cut videos into scene clips. 映射器将视频剪切成场景剪辑。 | [code](../data_juicer/ops/mapper/video_split_by_scene_mapper.py) | [tests](../tests/ops/mapper/test_video_split_by_scene_mapper.py) |
| video_tagging_from_audio_mapper | 🎬Video 💻CPU 🧩HF 🟢Stable | Mapper to generate video tags from audio streams extracted by video using the Audio Spectrogram Transformer. 映射器使用音频频谱图转换器从视频提取的音频流生成视频标签。 | [code](../data_juicer/ops/mapper/video_tagging_from_audio_mapper.py) | [tests](../tests/ops/mapper/test_video_tagging_from_audio_mapper.py) |
| video_tagging_from_frames_mapper | 🎬Video 💻CPU 🟢Stable | Mapper to generate video tags from frames extract by video. 映射器从视频提取的帧生成视频标签。 | [code](../data_juicer/ops/mapper/video_tagging_from_frames_mapper.py) | [tests](../tests/ops/mapper/test_video_tagging_from_frames_mapper.py) |
| whitespace_normalization_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to normalize different kinds of whitespaces to whitespace ' ' (0x20) in text samples. 映射器，用于将文本示例中的不同类型的空白标准化为空白 “” (0 x20)。 | [code](../data_juicer/ops/mapper/whitespace_normalization_mapper.py) | [tests](../tests/ops/mapper/test_whitespace_normalization_mapper.py) |

## selector <a name="selector"/>

| Operator 算子 | Tags 标签 | Description 描述 | Source code 源码 | Unit tests 单测样例 |
|----------|------|-------------|-------------|------------|
| frequency_specified_field_selector | 💻CPU 🟢Stable | Selector to select samples based on the sorted frequency of specified field. 选择器根据指定字段的排序频率选择样本。 | [code](../data_juicer/ops/selector/frequency_specified_field_selector.py) | [tests](../tests/ops/selector/test_frequency_specified_field_selector.py) |
| random_selector | 💻CPU 🟢Stable | Selector to random select samples. 选择器来随机选择样本。 | [code](../data_juicer/ops/selector/random_selector.py) | [tests](../tests/ops/selector/test_random_selector.py) |
| range_specified_field_selector | 💻CPU 🟢Stable | Selector to select a range of samples based on the sorted specified field value from smallest to largest. 选择器根据从最小到最大的排序指定字段值选择样本范围。 | [code](../data_juicer/ops/selector/range_specified_field_selector.py) | [tests](../tests/ops/selector/test_range_specified_field_selector.py) |
| tags_specified_field_selector | 💻CPU 🟢Stable | Selector to select samples based on the tags of specified field. 选择器根据指定字段的标签选择样本。 | [code](../data_juicer/ops/selector/tags_specified_field_selector.py) | [tests](../tests/ops/selector/test_tags_specified_field_selector.py) |
| topk_specified_field_selector | 💻CPU 🟢Stable | Selector to select top samples based on the sorted specified field value. 选择器根据已排序的指定字段值选择顶部样本。 | [code](../data_juicer/ops/selector/topk_specified_field_selector.py) | [tests](../tests/ops/selector/test_topk_specified_field_selector.py) |


## Contributing  贡献

We welcome contributions of adding new operators. Please refer to [How-to Guide
for Developers](DeveloperGuide.md).

我们欢迎社区贡献新的算子，具体请参考[开发者指南](DeveloperGuide_ZH.md)。
