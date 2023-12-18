import copy
import inspect
import os

from loguru import logger

DEFAULT_PREFIX = '__dj__'


class Fields(object):
    stats = DEFAULT_PREFIX + 'stats__'
    meta = DEFAULT_PREFIX + 'meta__'
    context = DEFAULT_PREFIX + 'context__'
    suffix = DEFAULT_PREFIX + 'suffix__'


class StatsKeysMeta(type):
    """
    a helper class to track the mapping from OP's name to its used stats_keys

    e.g., # once the AlphanumericFilter's compute_stats method has been called
    res = TrackingDescriptor.get_access_log()
    print(res) # {"AlphanumericFilter": ["alnum_ratio", "alpha_token_ratio"]}
    """
    _accessed_by = {}

    def __getattr__(cls, attr):
        caller_class = inspect.currentframe().f_back.f_globals['__name__']
        # no need to track the parent classes
        caller_class = caller_class.split('.')[-1]
        stat_key = getattr(cls._constants_class, attr)
        if caller_class not in cls._accessed_by:
            cls._accessed_by[caller_class] = set()
        if stat_key not in cls._accessed_by[caller_class]:
            cls._accessed_by[caller_class].add(stat_key)
        return stat_key

    def get_access_log(cls, dj_cfg=None):
        if cls._accessed_by:
            return cls._accessed_by
        elif dj_cfg:
            tmp_dj_cfg = copy.deepcopy(dj_cfg)
            # the access has been skipped due to the use of cache
            # we will using a temp data sample to get the access log
            if os.path.exists(dj_cfg.dataset_path) and \
                    'jsonl' in dj_cfg.dataset_path:
                logger.info(
                    'Begin to track the usage of ops with a dummy data sample')

                # load the first line as tmp_data
                tmp_f_name = dj_cfg.dataset_path.\
                    replace('.jsonl', '.tmp.jsonl')
                with open(dj_cfg.dataset_path, 'r') as orig_file, \
                        open(tmp_f_name, 'w') as tmp_file:
                    first_line = orig_file.readline()
                    tmp_file.write(first_line)

                tmp_dj_cfg.dataset_path = tmp_f_name
                tmp_dj_cfg.use_cache = False
                tmp_dj_cfg.use_checkpoint = False

                from data_juicer.core import Analyser
                tmp_analyzer = Analyser(tmp_dj_cfg)
                tmp_analyzer.run()

                os.remove(tmp_f_name)
            else:
                raise NotImplementedError(
                    f'For now, the dummy data is supported for only jsonl type'
                    f'. Please check your config as {dj_cfg.dataset_path} is '
                    f'either not existed or in jsonl type.')

        return cls._accessed_by


class StatsKeysConstant(object):
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


class StatsKeys(object, metaclass=StatsKeysMeta):
    _constants_class = StatsKeysConstant


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
