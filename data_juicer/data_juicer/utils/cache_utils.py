import os

# Default cache location
DEFAULT_CACHE_HOME = '~/.cache'
CACHE_HOME = os.getenv('CACHE_HOME', DEFAULT_CACHE_HOME)

# Default data_juicer cache location
DEFAULT_DATA_JUICER_CACHE_HOME = os.path.join(CACHE_HOME, 'data_juicer')
DATA_JUICER_CACHE_HOME = os.path.expanduser(
    os.getenv('DATA_JUICER_CACHE_HOME', DEFAULT_DATA_JUICER_CACHE_HOME))

# Default assets cache location
DEFAULT_DATA_JUICER_ASSETS_CACHE = os.path.join(DATA_JUICER_CACHE_HOME,
                                                'assets')
DATA_JUICER_ASSETS_CACHE = os.getenv('DATA_JUICER_ASSETS_CACHE',
                                     DEFAULT_DATA_JUICER_ASSETS_CACHE)
# Default models cache location
DEFAULT_DATA_JUICER_MODELS_CACHE = os.path.join(DATA_JUICER_CACHE_HOME,
                                                'models')
DATA_JUICER_MODELS_CACHE = os.getenv('DATA_JUICER_MODELS_CACHE',
                                     DEFAULT_DATA_JUICER_MODELS_CACHE)

CACHE_COMPRESS = None
