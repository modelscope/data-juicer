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

# Packages to corresponding ops that require them
PKG_TO_OPS = {
    'torch': [
        'image_aesthetics_filter', 'image_nsfw_filter',
        'image_text_matching_filter', 'image_text_similarity_filter',
        'image_watermark_filter', 'phrase_grounding_recall_filter',
        'video_aesthetics_filter', 'video_frames_text_similarity_filter',
        'video_nsfw_filter', 'video_tagging_from_frames_filter',
        'video_watermark_filter', 'generate_qa_from_text_mapper',
        'generate_qa_from_examples_mapper', 'image_captioning_mapper',
        'image_diffusion_mapper', 'image_tagging_mapper',
        'optimize_query_mapper', 'optimize_response_mapper',
        'optimize_qa_mapper', 'video_captioning_from_frames_mapper',
        'video_captioning_from_summarizer_mapper',
        'video_captioning_from_video_mapper',
        'video_tagging_from_audio_mapper', 'video_tagging_from_frames_mapper'
    ],
    'torchaudio': [
        'video_captioning_from_summarizer_mapper',
        'video_tagging_from_audio_mapper'
    ],
    'easyocr': ['video_ocr_area_ratio_filter'],
    'fasttext-wheel': ['language_id_score_filter'],
    'kenlm': ['perplexity_filter'],
    'sentencepiece': [
        'flagged_words_filter', 'perplexity_filter', 'stopwords_filter',
        'word_repetition_filter', 'words_num_filter'
    ],
    'scipy': ['document_minhash_deduplicator'],
    'ftfy': ['fix_unicode_mapper'],
    'simhash-pybind': [
        'document_simhash_deduplicator', 'image_captioning_mapper',
        'image_diffusion_mapper', 'video_captioning_from_frames_mapper',
        'video_captioning_from_summarizer_mapper',
        'video_captioning_from_video_mapper'
    ],
    'selectolax': ['clean_html_mapper'],
    'nlpaug': ['nlpaug_en_mapper'],
    'nlpcda': ['nlpcda'],
    'nltk': ['phrase_grounding_recall_filter', 'sentence_split_mapper'],
    'transformers': [
        'alphanumeric_filter', 'image_aesthetics_filter', 'image_nsfw_filter',
        'image_text_matching_filter', 'image_text_similarity_filter',
        'image_watermark_filter', 'phrase_grounding_recall_filter',
        'token_num_filter', 'video_aesthetics_filter',
        'video_frames_text_similarity_filter', 'video_nsfw_filter',
        'generate_qa_from_text_mapper', 'generate_qa_from_examples_mapper',
        'image_captioning_mapper', 'image_diffusion_mapper',
        'optimize_query_mapper', 'optimize_response_mapper',
        'optimize_qa_mapper', 'video_captioning_from_audio_mapper',
        'video_captioning_from_frames_mapper',
        'video_captioning_from_summarizer_mapper',
        'video_captioning_from_video_mapper', 'video_tagging_from_audio_mapper'
    ],
    'transformers_stream_generator': [
        'video_captioning_from_audio_mapper',
        'video_captioning_from_summarizer_mapper'
    ],
    'einops': [
        'video_captioning_from_audio_mapper',
        'video_captioning_from_summarizer_mapper'
    ],
    'accelerate': [
        'video_captioning_from_audio_mapper',
        'video_captioning_from_summarizer_mapper'
    ],
    'tiktoken': [
        'video_captioning_from_audio_mapper',
        'video_captioning_from_summarizer_mapper'
    ],
    'opencc': ['chinese_convert_mapper'],
    'imagededup': ['image_deduplicator', 'ray_image_deduplicator'],
    'spacy-pkuseg': ['text_action_filter', 'text_entity_dependency_filter'],
    'diffusers': ['image_diffusion_mapper'],
    'simple-aesthetics-predictor':
    ['image_aesthetics_filter', 'video_aesthetics_filter'],
    'scenedetect[opencv]': ['video_split_by_scene_mapper'],
    'ffmpeg-python': [
        'audio_ffmpeg_wrapped_mapper', 'video_ffmpeg_wrapped_mapper',
        'video_resize_aspect_ratio_mapper', 'video_resize_resolution_mapper'
    ],
    'opencv-python': [
        'image_face_ratio_filter', 'video_motion_score_filter',
        'image_face_blur_mapper', 'video_face_blur_mapper',
        'video_remove_watermark_mapper'
    ],
    'vllm': [
        'generate_qa_from_text_mapper',
        'generate_qa_from_examples_mapper',
        'optimize_query_mapper',
        'optimize_response_mapper',
        'optimize_qa_mapper',
    ],
    'rouge': ['generate_qa_from_examples_mapper'],
    'ram': ['image_tagging_mapper', 'video_tagging_from_frames_mapper']
}
