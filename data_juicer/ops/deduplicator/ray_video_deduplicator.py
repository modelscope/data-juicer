import hashlib

from pydantic import PositiveInt

from data_juicer.utils.mm_utils import (close_video, load_data_with_context,
                                        load_video)

from ..base_op import OPERATORS
from ..op_fusion import LOADED_VIDEOS
from .ray_basic_deduplicator import RayBasicDeduplicator

OP_NAME = 'ray_video_deduplicator'


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class RayVideoDeduplicator(RayBasicDeduplicator):
    """
    Deduplicator to deduplicate samples at document-level using exact matching
    of videos between documents.
    """

    def __init__(self,
                 redis_host: str = 'localhost',
                 redis_port: PositiveInt = 6380,
                 *args,
                 **kwargs):
        """
        Initialization.
        :param redis_host: the hostname of redis server
        :param redis_port: the port of redis server
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(redis_host=redis_host,
                         redis_port=redis_port,
                         *args,
                         **kwargs)

    def calculate_hash(self, sample, context=False):
        if self.video_key not in sample or not sample[self.video_key]:
            return RayBasicDeduplicator.EMPTY_HASH_VALUE

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)
        # compute hash
        md5_hash = hashlib.md5()
        for key in videos:
            # consider the multi stream of video in one container
            for packet in videos[key].demux():
                if packet.stream.type == 'video':
                    md5_hash.update(bytes(packet))

        for key in videos:
            close_video(videos[key])

        return md5_hash.hexdigest()
