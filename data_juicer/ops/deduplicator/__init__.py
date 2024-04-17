from . import (document_deduplicator, document_minhash_deduplicator,
               document_simhash_deduplicator, image_deduplicator,
               ray_document_deduplicator, ray_image_deduplicator,
               ray_video_deduplicator, video_deduplicator)
from .document_deduplicator import DocumentDeduplicator
from .document_minhash_deduplicator import DocumentMinhashDeduplicator
from .document_simhash_deduplicator import DocumentSimhashDeduplicator
from .image_deduplicator import ImageDeduplicator
from .ray_basic_deduplicator import RayBasicDeduplicator
from .ray_document_deduplicator import RayDocumentDeduplicator
from .ray_image_deduplicator import RayImageDeduplicator
from .ray_video_deduplicator import RayVideoDeduplicator
from .video_deduplicator import VideoDeduplicator

__all__ = [
    'VideoDeduplicator', 'RayBasicDeduplicator', 'DocumentMinhashDeduplicator',
    'RayImageDeduplicator', 'RayDocumentDeduplicator', 'DocumentDeduplicator',
    'ImageDeduplicator', 'DocumentSimhashDeduplicator', 'RayVideoDeduplicator'
]
