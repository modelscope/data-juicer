# Map the imported module to the require package we need to install
# keep sorted for maintainability
MODULE_TO_PKGS = {
    'PIL': ['Pillow'],
    'aesthetics_predictor': ['simple-aesthetics-predictor'],
    'cv2': ['opencv-python'],
    'fasttext': ['fasttext-wheel'],
    'ffmpeg': ['ffmpeg-python'],
    'ram': ['ram@git+https://github.com/xinyu1205/recognize-anything.git'],
    'scenedetect': ['scenedetect[opencv]'],
    'simhash': ['simhash-pybind']
}

# Packages to corresponding ops that require them
# keep sorted for maintainability
PKG_TO_OPS = {
    'accelerate': [
        'video_captioning_from_audio_mapper',
        'video_captioning_from_summarizer_mapper'
    ],
    'diffusers': ['image_diffusion_mapper'],
    'easyocr': ['video_ocr_area_ratio_filter'],
    'einops': [
        'video_captioning_from_audio_mapper',
        'video_captioning_from_summarizer_mapper'
    ],
    'fasttext-wheel': ['language_id_score_filter'],
    'ffmpeg-python': [
        'audio_ffmpeg_wrapped_mapper', 'video_ffmpeg_wrapped_mapper',
        'video_resize_aspect_ratio_mapper', 'video_resize_resolution_mapper'
    ],
    'ftfy': ['fix_unicode_mapper'],
    'imagededup': ['image_deduplicator', 'ray_image_deduplicator'],
    'kenlm': ['perplexity_filter'],
    'nlpaug': ['nlpaug_en_mapper'],
    'nlpcda': ['nlpcda'],
    'nltk': ['phrase_grounding_recall_filter', 'sentence_split_mapper'],
    'opencc': ['chinese_convert_mapper'],
    'opencv-python': [
        'image_face_blur_mapper', 'image_face_ratio_filter',
        'video_face_blur_mapper', 'video_motion_score_filter',
        'video_remove_watermark_mapper'
    ],
    'ram': ['image_tagging_mapper', 'video_tagging_from_frames_mapper'],
    'rouge': ['generate_qa_from_examples_mapper'],
    'scenedetect[opencv]': ['video_split_by_scene_mapper'],
    'scipy': ['document_minhash_deduplicator'],
    'selectolax': ['clean_html_mapper'],
    'sentencepiece': [
        'flagged_words_filter', 'perplexity_filter', 'stopwords_filter',
        'word_repetition_filter', 'words_num_filter'
    ],
    'simhash-pybind': [
        'document_simhash_deduplicator', 'image_captioning_mapper',
        'image_diffusion_mapper', 'video_captioning_from_frames_mapper',
        'video_captioning_from_summarizer_mapper',
        'video_captioning_from_video_mapper'
    ],
    'simple-aesthetics-predictor':
    ['image_aesthetics_filter', 'video_aesthetics_filter'],
    'spacy-pkuseg': ['text_action_filter', 'text_entity_dependency_filter'],
    'tiktoken': [
        'video_captioning_from_audio_mapper',
        'video_captioning_from_summarizer_mapper'
    ],
    'torch': [
        'generate_qa_from_examples_mapper', 'generate_qa_from_text_mapper',
        'image_aesthetics_filter', 'image_captioning_mapper',
        'image_diffusion_mapper', 'image_nsfw_filter', 'image_segment_mapper',
        'image_tagging_mapper', 'image_text_matching_filter',
        'image_text_similarity_filter', 'image_watermark_filter',
        'optimize_qa_mapper', 'optimize_query_mapper',
        'optimize_response_mapper', 'phrase_grounding_recall_filter',
        'video_aesthetics_filter', 'video_captioning_from_frames_mapper',
        'video_captioning_from_summarizer_mapper',
        'video_captioning_from_video_mapper',
        'video_frames_text_similarity_filter', 'video_nsfw_filter',
        'video_tagging_from_audio_mapper', 'video_tagging_from_frames_filter',
        'video_tagging_from_frames_mapper', 'video_watermark_filter'
    ],
    'torchaudio': [
        'video_captioning_from_summarizer_mapper',
        'video_tagging_from_audio_mapper'
    ],
    'transformers': [
        'alphanumeric_filter', 'generate_qa_from_examples_mapper',
        'generate_qa_from_text_mapper', 'image_aesthetics_filter',
        'image_captioning_mapper', 'image_diffusion_mapper',
        'image_nsfw_filter', 'image_text_matching_filter',
        'image_text_similarity_filter', 'image_watermark_filter',
        'optimize_qa_mapper', 'optimize_query_mapper',
        'optimize_response_mapper', 'phrase_grounding_recall_filter',
        'token_num_filter', 'video_aesthetics_filter',
        'video_captioning_from_audio_mapper',
        'video_captioning_from_frames_mapper',
        'video_captioning_from_summarizer_mapper',
        'video_captioning_from_video_mapper',
        'video_frames_text_similarity_filter', 'video_nsfw_filter',
        'video_tagging_from_audio_mapper'
    ],
    'transformers_stream_generator': [
        'video_captioning_from_audio_mapper',
        'video_captioning_from_summarizer_mapper'
    ],
    'ultralytics': ['image_segment_mapper'],
    'vllm': [
        'generate_qa_from_examples_mapper', 'generate_qa_from_text_mapper',
        'optimize_qa_mapper', 'optimize_query_mapper',
        'optimize_response_mapper'
    ]
}
