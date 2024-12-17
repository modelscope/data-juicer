from . import deduplicator, filter, mapper, selector
from .base_op import (OPERATORS, UNFORKABLE, Deduplicator, Filter, Mapper,
                      Selector)
from .load import load_ops

__all__ = [
    'load_ops',
    'Filter',
    'Mapper',
    'Deduplicator',
    'Selector',
    'NON_STATS_FILTERS',
    'TAGGING_OPS',
]

# Filters that don't produce any stats
NON_STATS_FILTERS = {
    'specified_field_filter',
    'specified_numeric_field_filter',
    'suffix_filter',
    'video_tagging_from_frames_filter',
}

# OPs that will produce tags in meta
TAGGING_OPS = {
    'video_tagging_from_frames_filter',
    'video_tagging_from_audio_mapper',
    'image_tagging_mapper',
}
