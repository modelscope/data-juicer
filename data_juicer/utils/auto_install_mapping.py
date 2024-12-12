# Map the imported module to the require package we need to install
MODULE_TO_PKGS = {
    'aesthetics_predictor': ['simple-aesthetics-predictor'],
    'cv2': ['opencv-python'],
    'fasttext': ['fasttext-wheel'],
    'ffmpeg': ['ffmpeg-python'],
    'PIL': ['Pillow'],
    'ram': ['ram@git+https://github.com/xinyu1205/recognize-anything.git'],
    'scenedetect': ['scenedetect[opencv]'],
    'simhash': ['simhash-pybind'],
}

# Extra packages required by each op
OPS_TO_PKG = {
    'video_aesthetics_filter':
    ['simple-aesthetics-predictor', 'torch', 'transformers'],
    'document_simhash_deduplicator': ['simhash-pybind'],
    'nlpcda_zh_mapper': ['nlpcda'],
    'image_aesthetics_filter':
    ['simple-aesthetics-predictor', 'torch', 'transformers'],
    'video_nsfw_filter': ['torch', 'transformers'],
    'video_face_blur_mapper': ['opencv-python'],
    'stopwords_filter': ['sentencepiece'],
    'fix_unicode_mapper': ['ftfy'],
    'token_num_filter': ['transformers'],
    'optimize_qa_mapper': ['torch', 'transformers', 'vllm'],
    'video_motion_score_filter': ['opencv-python'],
    'image_tagging_mapper': ['ram', 'torch'],
    'video_resize_aspect_ratio_mapper': ['ffmpeg-python'],
    'video_captioning_from_audio_mapper': [
        'accelerate', 'einops', 'tiktoken', 'transformers',
        'transformers_stream_generator'
    ],
    'clean_html_mapper': ['selectolax'],
    'video_tagging_from_audio_mapper': ['torch', 'torchaudio', 'transformers'],
    'image_deduplicator': ['imagededup'],
    'image_diffusion_mapper':
    ['diffusers', 'simhash-pybind', 'torch', 'transformers'],
    'image_text_similarity_filter': ['torch', 'transformers'],
    'alphanumeric_filter': ['transformers'],
    'image_nsfw_filter': ['torch', 'transformers'],
    'image_watermark_filter': ['torch', 'transformers'],
    'ray_image_deduplicator': ['imagededup'],
    'video_captioning_from_frames_mapper':
    ['simhash-pybind', 'torch', 'transformers'],
    'video_tagging_from_frames_filter': ['torch'],
    'video_resize_resolution_mapper': ['ffmpeg-python'],
    'optimize_query_mapper': ['torch', 'transformers', 'vllm'],
    'sentence_split_mapper': ['nltk'],
    'image_text_matching_filter': ['torch', 'transformers'],
    'phrase_grounding_recall_filter': ['nltk', 'torch', 'transformers'],
    'video_split_by_scene_mapper': ['scenedetect[opencv]'],
    'image_face_blur_mapper': ['opencv-python'],
    'image_face_ratio_filter': ['opencv-python'],
    'document_minhash_deduplicator': ['scipy'],
    'flagged_words_filter': ['sentencepiece'],
    'language_id_score_filter': ['fasttext-wheel'],
    'words_num_filter': ['sentencepiece'],
    'chinese_convert_mapper': ['opencc'],
    'video_frames_text_similarity_filter': ['torch', 'transformers'],
    'generate_qa_from_text_mapper': ['torch', 'transformers', 'vllm'],
    'video_ffmpeg_wrapped_mapper': ['ffmpeg-python'],
    'image_captioning_mapper': ['simhash-pybind', 'torch', 'transformers'],
    'video_ocr_area_ratio_filter': ['easyocr'],
    'video_captioning_from_video_mapper':
    ['simhash-pybind', 'torch', 'transformers'],
    'video_remove_watermark_mapper': ['opencv-python'],
    'text_action_filter': ['spacy-pkuseg'],
    'nlpaug_en_mapper': ['nlpaug'],
    'word_repetition_filter': ['sentencepiece'],
    'video_watermark_filter': ['torch'],
    'video_captioning_from_summarizer_mapper': [
        'accelerate', 'einops', 'simhash-pybind', 'tiktoken', 'torch',
        'torchaudio', 'transformers', 'transformers_stream_generator'
    ],
    'audio_ffmpeg_wrapped_mapper': ['ffmpeg-python'],
    'perplexity_filter': ['kenlm', 'sentencepiece'],
    'generate_qa_from_examples_mapper':
    ['rouge', 'torch', 'transformers', 'vllm'],
    'video_tagging_from_frames_mapper': ['ram', 'torch'],
    'text_entity_dependency_filter': ['spacy-pkuseg'],
    'optimize_response_mapper': ['torch', 'transformers', 'vllm']
}
