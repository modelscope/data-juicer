from data_juicer.utils.lazy_loader import LazyLoader

from .document_deduplicator import DocumentDeduplicator
from .document_minhash_deduplicator import DocumentMinhashDeduplicator
from .document_simhash_deduplicator import DocumentSimhashDeduplicator
from .image_deduplicator import ImageDeduplicator
from .video_deduplicator import VideoDeduplicator

RayBasicDeduplicator = LazyLoader(
    'RayBasicDeduplicator',
    'data_juicer.ops.deduplicator.ray_basic_deduplicator.RayBasicDeduplicator',
    auto_install=False)
RayBTSMinhashDeduplicator = LazyLoader(
    'RayBTSMinhashDeduplicator',
    'data_juicer.ops.deduplicator.ray_bts_minhash_deduplicator.'
    'RayBTSMinhashDeduplicator',
    auto_install=False)
RayDocumentDeduplicator = LazyLoader(
    'RayDocumentDeduplicator',
    'data_juicer.ops.deduplicator.ray_document_deduplicator.'
    'RayDocumentDeduplicator',
    auto_install=False)
RayImageDeduplicator = LazyLoader(
    'RayImageDeduplicator',
    'data_juicer.ops.deduplicator.ray_image_deduplicator.RayImageDeduplicator',
    auto_install=False)
RayVideoDeduplicator = LazyLoader(
    'RayVideoDeduplicator',
    'data_juicer.ops.deduplicator.ray_video_deduplicator.RayVideoDeduplicator',
    auto_install=False)

__all__ = [
    'DocumentDeduplicator',
    'DocumentMinhashDeduplicator',
    'DocumentSimhashDeduplicator',
    'ImageDeduplicator',
    'RayBasicDeduplicator',
    'RayDocumentDeduplicator',
    'RayImageDeduplicator',
    'RayVideoDeduplicator',
    'RayImageDeduplicator',
    'RayBTSMinhashDeduplicator',
    'VideoDeduplicator',
]
