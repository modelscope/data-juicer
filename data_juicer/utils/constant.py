DEFAULT_PREFIX = '__dj__'


class Fields(object):
    stats = DEFAULT_PREFIX + 'stats__'
    meta = DEFAULT_PREFIX + 'meta__'
    context = DEFAULT_PREFIX + 'context__'
    suffix = DEFAULT_PREFIX + 'suffix__'


class StatsKeys(object):
    # text
    alpha_token_ratio = 'alpha_token_ratio'
    alnum_ratio = 'alnum_ratio'
    avg_line_length = 'avg_line_length'
    char_rep_ratio = 'char_rep_ratio'
    flagged_words_ratio = 'flagged_words_ratio'
    lang = 'lang'
    lang_score = 'lang_score'
    max_line_length = 'max_line_length'
    perplexity = 'perplexity'
    special_char_ratio = 'special_char_ratio'
    stopwords_ratio = 'stopwords_ratio'
    text_len = 'text_len'
    num_action = 'num_action'
    num_dependency_edges = 'num_dependency_edges'
    num_token = 'num_token'
    num_words = 'num_words'
    word_rep_ratio = 'word_rep_ratio'

    # image
    aspect_ratios = 'aspect_ratios'
    image_width = 'image_width'
    image_height = 'image_height'
    image_sizes = 'image_sizes'
    face_ratios = 'face_ratios'
    face_detections = 'face_detections'

    # multimodal
    image_text_similarity = 'image_text_similarity'
    image_text_matching_score = 'image_text_matching_score'


class HashKeys(object):
    hash = DEFAULT_PREFIX + 'hash'
    minhash = DEFAULT_PREFIX + 'minhash'
    simhash = DEFAULT_PREFIX + 'simhash'

    # image
    imagehash = DEFAULT_PREFIX + 'imagehash'


class InterVars(object):
    # text
    lines = DEFAULT_PREFIX + 'lines'
    words = DEFAULT_PREFIX + 'words'
    refined_words = DEFAULT_PREFIX + 'refined_words'

    # image
    loaded_images = DEFAULT_PREFIX + 'loaded_images'
