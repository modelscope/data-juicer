# Operator Schemas

Operators are a collection of basic processes that assist in data modification, cleaning, filtering, deduplication, etc. We support a wide range of data sources and file formats, and allow for flexible extension to custom datasets.

This page offers a basic description of the operators (OPs) in Data-Juicer. Users can refer to the [API documentation](https://modelscope.github.io/data-juicer/) for the specific parameters of each operator. Users can refer to and run the unit tests (`tests/ops/...`) for [examples of operator-wise usage](../tests/ops) as well as the effects of each operator when applied to built-in test data samples.

## Overview

The operators in Data-Juicer are categorized into 5 types.

| Type                              | Number | Description                                     |
|-----------------------------------|:------:|-------------------------------------------------|
| [ Formatter ]( #formatter )       |   7    | Discovers, loads, and canonicalizes source data |
| [ Mapper ]( #mapper )             |   47   | Edits and transforms samples                    |
| [ Filter ]( #filter )             |   42   | Filters out low-quality samples                 |
| [ Deduplicator ]( #deduplicator ) |   5    | Detects and removes duplicate samples           |
| [ Selector ]( #selector )         |   4    | Selects top samples based on ranking            |


All the specific operators are listed below, each featured with several capability tags.

* Domain Tags
    - General: general purpose
    - LaTeX: specific to LaTeX source files
    - Code: specific to programming codes
    - Financial: closely related to financial sector
    - Image: specific to images or multimodal
    - Audio: specific to audios or multimodal
    - Video: specific to videos or multimodal
    - Multimodal: specific to multimodal
* Language Tags
    - en: English
    - zh: Chinese


## Formatter <a name="formatter"/>

| Operator          | Domain  |  Lang  | Description                                                                                       |
|-------------------|---------|--------|---------------------------------------------------------------------------------------------------|
| remote_formatter  | General | en, zh | Prepares datasets from remote (e.g., HuggingFace)                                                 |
| csv_formatter     | General | en, zh | Prepares local `.csv` files                                                                       |
| tsv_formatter     | General | en, zh | Prepares local `.tsv` files                                                                       |
| json_formatter    | General | en, zh | Prepares local `.json`, `.jsonl`, `.jsonl.zst` files                                              |
| parquet_formatter | General | en, zh | Prepares local `.parquet` files                                                                   |
| text_formatter    | General | en, zh | Prepares other local text files ([complete list](../data_juicer/format/text_formatter.py#L63,73)) |
| mixture_formatter | General | en, zh | Handles a mixture of all the supported local file types                                           | 


## Mapper <a name="mapper"/>

| Operator                                            | Domain             | Lang   | Description                                                                                                                                                                    |
|-----------------------------------------------------|--------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| audio_ffmpeg_wrapped_mapper                         | Audio              | -      | Simple wrapper to run a FFmpeg audio filter                                                                                                                                    |
| chinese_convert_mapper                              | General            | zh     | Converts Chinese between Traditional Chinese, Simplified Chinese and Japanese Kanji (by [opencc](https://github.com/BYVoid/OpenCC))                                            |
| clean_copyright_mapper                              | Code               | en, zh | Removes copyright notice at the beginning of code files (must contain the word *copyright*)                                                                                    |
| clean_email_mapper                                  | General            | en, zh | Removes email information                                                                                                                                                      |
| clean_html_mapper                                   | General            | en, zh | Removes HTML tags and returns plain text of all the nodes                                                                                                                      |
| clean_ip_mapper                                     | General            | en, zh | Removes IP addresses                                                                                                                                                           |
| clean_links_mapper                                  | General, Code      | en, zh | Removes links, such as those starting with http or ftp                                                                                                                         |
| expand_macro_mapper                                 | LaTeX              | en, zh | Expands macros usually defined at the top of TeX documents                                                                                                                     |
| extract_qa_mapper                                   | General            | en, zh | Extract question and answer pair from text samples.                                                                                                                            |
| fix_unicode_mapper                                  | General            | en, zh | Fixes broken Unicodes (by [ftfy](https://ftfy.readthedocs.io/))                                                                                                                |
| generate_instruction_mapper                         | General            | en, zh | Generate instruction text samples.                                                                                                                                             |
| image_blur_mapper                                   | Image              |  -     | Blur images                                                                                                                                                                    |
| image_captioning_from_gpt4v_mapper                  | Multimodal         |  -     | generate samples whose texts are generated based on gpt-4-visison and the image                                                                                                |
| image_captioning_mapper                             | Multimodal         |  -     | generate samples whose captions are generated based on another model (such as blip2) and the figure within the original sample                                                 |
| image_diffusion_mapper                              | Multimodal         |  -     | Generate and augment images by stable diffusion model                                                                                                                          |
| image_face_blur_mapper                              | Image              |  -     | Blur faces detected in images                                                                                                                                                  |
| image_tagging_mapper                    | Multimodal         | -      | Mapper to generate image tags from the input images.                                                                                                                           |
| nlpaug_en_mapper                                    | General            | en     | Simply augments texts in English based on the `nlpaug` library                                                                                                                 | 
| nlpcda_zh_mapper                                    | General            | zh     | Simply augments texts in Chinese based on the `nlpcda` library                                                                                                                 | 
| optimize_instruction_mapper                         | General            | en, zh | Optimize instruction text samples.                                                                                                                                             |
| punctuation_normalization_mapper                    | General            | en, zh | Normalizes various Unicode punctuations to their ASCII equivalents                                                                                                             |
| remove_bibliography_mapper                          | LaTeX              | en, zh | Removes the bibliography of TeX documents                                                                                                                                      |
| remove_comments_mapper                              | LaTeX              | en, zh | Removes the comments of TeX documents                                                                                                                                          |
| remove_header_mapper                                | LaTeX              | en, zh | Removes the running headers of TeX documents, e.g., titles, chapter or section numbers/names                                                                                   |
| remove_long_words_mapper                            | General            | en, zh | Removes words with length outside the specified range                                                                                                                          |
| remove_non_chinese_character_mapper                 | General            | en, zh | Remove non Chinese character in text samples.                                                                                                                                  |
| remove_repeat_sentences_mapper                      | General            | en, zh | Remove repeat sentences in text samples.                                                                                                                                       |
| remove_specific_chars_mapper                        | General            | en, zh | Removes any user-specified characters or substrings                                                                                                                            |
| remove_table_text_mapper                            | General, Financial | en     | Detects and removes possible table contents (:warning: relies on regular expression matching and thus fragile)                                                                 |
| remove_words_with_incorrect_<br />substrings_mapper | General            | en, zh | Removes words containing specified substrings                                                                                                                                  |
| replace_content_mapper                              | General            | en, zh | Replace all content in the text that matches a specific regular expression pattern with a designated replacement string                                                        |
| sentence_split_mapper                               | General            | en     | Splits and reorganizes sentences according to semantics                                                                                                                        |
| video_captioning_from_audio_mapper                  | Multimodal         | -      | Caption a video according to its audio streams based on Qwen-Audio model                                                                                                       |
| video_captioning_from_frames_mapper                 | Multimodal         |  -     | generate samples whose captions are generated based on an image-to-text model and sampled video frames. Captions from different frames will be concatenated to a single string |
| video_captioning_from_summarizer_mapper             | Multimodal         | -      | Generate video captions by summarizing several kinds of generated texts (captions from video/audio/frames, tags from audio/frames, ...)                                        |
| video_captioning_from_video_mapper                  | Multimodal         |  -     | generate samples whose captions are generated based on another model (video-blip) and sampled video frame within the original sample                                           |
| video_face_blur_mapper                              | Video              |  -     | Blur faces detected in videos                                                                                                                                                  |
| video_ffmpeg_wrapped_mapper                         | Video              | -      | Simple wrapper to run a FFmpeg video filter                                                                                                                                    |
| video_remove_watermark_mapper                       | Video              | -      | Remove the watermarks in videos given regions                                                                                                                                  |
| video_resize_aspect_ratio_mapper                    | Video              | -      | Resize video aspect ratio to a specified range                                                                                                                                 |
| video_resize_resolution_mapper                      | Video              | -      | Map videos to ones with given resolution range                                                                                                                                 |
| video_split_by_duration_mapper                      | Multimodal         | -      | Mapper to split video by duration                                                                                                                                              |
| video_spit_by_key_frame_mapper                      | Multimodal         | -      | Mapper to split video by key frame                                                                                                                                             |
| video_split_by_scene_mapper                         | Multimodal         | -      | Split videos into scene clips                                                                                                                                                  |
| video_tagging_from_audio_mapper                     | Multimodal         | -      | Mapper to generate video tags from audio streams extracted from the video.                                                                                                     |
| video_tagging_from_frames_mapper                    | Multimodal         | -      | Mapper to generate video tags from frames extracted from the video.                                                                                                            |
| whitespace_normalization_mapper                     | General            | en, zh | Normalizes various Unicode whitespaces to the normal ASCII space (U+0020)                                                                                                      |


## Filter <a name="filter"/>

| Operator                       | Domain     | Lang   | Description                                                                                                                                         |
|--------------------------------|------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| alphanumeric_filter            | General    | en, zh | Keeps samples with alphanumeric ratio within the specified range                                                                                    |
| audio_duration_filter          | Audio      | -      | Keep data samples whose audios' durations are within a specified range                                                                              |
| audio_nmf_snr_filter           | Audio      | -      | Keep data samples whose audios' Signal-to-Noise Ratios (SNRs, computed based on Non-Negative Matrix Factorization, NMF) are within a specified range|
| audio_size_filter              | Audio      | -      | Keep data samples whose audios' sizes are within a specified range                                                                                  |
| average_line_length_filter     | Code       | en, zh | Keeps samples with average line length within the specified range                                                                                   |
| character_repetition_filter    | General    | en, zh | Keeps samples with char-level n-gram repetition ratio within the specified range                                                                    |
| flagged_words_filter           | General    | en, zh | Keeps samples with flagged-word ratio below the specified threshold                                                                                 |
| image_aesthetics_filter        | Image      | -      | Keeps samples containing images whose aesthetics scores are within the specified range                                                              |
| image_aspect_ratio_filter      | Image      | -      | Keeps samples containing images with aspect ratios within the specified range                                                                       |
| image_face_ratio_filter        | Image      | -      | Keeps samples containing images with face area ratios within the specified range                                                                    |
| image_nsfw_filter              | Image      | -      | Keeps samples containing images with NSFW scores below the threshold                                                               |
| image_pair_similarity_filter   | Image      | -      | Keeps image pairs with image feature cosine similarity within the specified range based on a CLIP model                                                    |
| image_shape_filter             | Image      | -      | Keeps samples containing images with widths and heights within the specified range                                                                  |
| image_size_filter              | Image      | -      | Keeps samples containing images whose size in bytes are within the specified range                                                                  |
| image_text_matching_filter     | Multimodal | -      | Keeps samples with image-text classification matching score within the specified range based on a BLIP model                                        |
| image_text_similarity_filter   | Multimodal | -      | Keeps samples with image-text feature cosine similarity within the specified range based on a CLIP model                                            |
| image_watermark_filter         | Image      | -      | Keeps samples containing images with predicted watermark probabilities below the threshold                                                               |
| language_id_score_filter       | General    | en, zh | Keeps samples of the specified language, judged by a predicted confidence score                                                                     |
| maximum_line_length_filter     | Code       | en, zh | Keeps samples with maximum line length within the specified range                                                                                   |
| perplexity_filter              | General    | en, zh | Keeps samples with perplexity score below the specified threshold                                                                                   |
| phrase_grounding_recall_filter | Multimodal | -      | Keeps samples whose locating recalls of phrases extracted from text in the images are within a specified range                                      |
| special_characters_filter      | General    | en, zh | Keeps samples with special-char ratio within the specified range                                                                                    |
| specified_field_filter         | General    | en, zh | Filters samples based on field, with value lies in the specified targets                                                                            |
| specified_numeric_field_filter | General    | en, zh | Filters samples based on field, with value lies in the specified range (for numeric types)                                                          |
| stopwords_filter               | General    | en, zh | Keeps samples with stopword ratio above the specified threshold                                                                                     |
| suffix_filter                  | General    | en, zh | Keeps samples with specified suffixes                                                                                                               |
| text_action_filter             | General    | en, zh | Keeps samples containing action verbs in their texts                                                                                                |
| text_entity_dependency_filter  | General    | en, zh | Keeps samples containing dependency edges for an entity in the dependency tree of the texts                                                   |
| text_length_filter             | General    | en, zh | Keeps samples with total text length within the specified range                                                                                     |
| token_num_filter               | General    | en, zh | Keeps samples with token count within the specified range                                                                                           |
| video_aesthetics_filter        | Video      | -      | Keeps samples whose specified frames have aesthetics scores within the specified range     |
| video_aspect_ratio_filter      | Video      | -      | Keeps samples containing videos with aspect ratios within the specified range                                                                       |
| video_duration_filter          | Video      | -      | Keep data samples whose videos' durations are within a specified range |                                                                            
| video_frames_text_similarity_filter    | Multimodal | -      | Keep data samples whose similarities between sampled video frame images and text are within a specific range |
| video_motion_score_filter      | Video      | -      | Keep samples with video motion scores within a specific range |
| video_nsfw_filter              | Video      | -      | Keeps samples containing videos with NSFW scores below the threshold                                                               |
| video_ocr_area_ratio_filter    | Video      | -      | Keep data samples whose detected text area ratios for specified frames in the video are within a specified range |                                  
| video_resolution_filter        | Video      | -      | Keeps samples containing videos with horizontal and vertical resolutions within the specified range                                                 |
| video_watermark_filter         | Video      | -      | Keeps samples containing videos with predicted watermark probabilities below the threshold                                                               |
| video_tagging_from_frames_filter  | Video   | -      | Keep samples containing videos with given tags |
| words_num_filter               | General    | en, zh | Keeps samples with word count within the specified range                                                                                            |
| word_repetition_filter         | General    | en, zh | Keeps samples with word-level n-gram repetition ratio within the specified range                                                                    |


## Deduplicator <a name="deduplicator"/>

| Operator                      | Domain  | Lang   | Description                                                  |
|-------------------------------|---------|--------|--------------------------------------------------------------|
| document_deduplicator         | General | en, zh | Deduplicates samples at document-level by comparing MD5 hash |
| document_minhash_deduplicator | General | en, zh | Deduplicates samples at document-level using MinHashLSH      |
| document_simhash_deduplicator | General | en, zh | Deduplicates samples at document-level using SimHash         |
| image_deduplicator            | Image   |   -    | Deduplicates samples at document-level using exact matching of images between documents |
| video_deduplicator            | Video   |   -    | Deduplicates samples at document-level using exact matching of videos between documents |
| ray_document_deduplicator     | General | en, zh | Deduplicates samples at document-level by comparing MD5 hash on ray                     |
| ray_image_deduplicator        | Image   |   -    | Deduplicates samples at document-level using exact matching of images between documents on ray |
| ray_video_deduplicator        | Video   |   -    | Deduplicates samples at document-level using exact matching of videos between documents on ray |

## Selector <a name="selector"/>

| Operator                           | Domain  | Lang   | Description                                                           |
|------------------------------------|---------|--------|-----------------------------------------------------------------------|
| frequency_specified_field_selector | General | en, zh | Selects top samples by comparing the frequency of the specified field |
| random_selector                    | General | en, zh | Selects samples randomly                                              |
| range_specified_field_selector     | General | en, zh | Selects samples within a specified range by comparing the values of the specified field    |
| topk_specified_field_selector      | General | en, zh | Selects top samples by comparing the values of the specified field    |


## Contributing
We welcome contributions of adding new operators. Please refer to [How-to Guide for Developers](DeveloperGuide.md).
