import numpy as np
from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS
from ..op_fusion import LOADED_IMAGES
from .ray_basic_deduplicator import RayBasicDeduplicator

OP_NAME = 'ray_image_deduplicator'

with AvailabilityChecking(['imagededup'], OP_NAME):
    import imagededup  # noqa: F401

    HASH_METHOD = {'phash', 'dhash', 'whash', 'ahash'}

    def get_hash_method(method_name):
        from imagededup.methods import AHash, DHash, PHash, WHash

        mapping = {
            'phash': PHash,
            'dhash': DHash,
            'whash': WHash,
            'ahash': AHash
        }

        return mapping[method_name]


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class RayImageDeduplicator(RayBasicDeduplicator):
    """
    Deduplicator to deduplicate samples at document-level using exact matching
    of images between documents.
    """

    def __init__(self,
                 redis_host: str = 'localhost',
                 redis_port: PositiveInt = 6380,
                 method: str = 'phash',
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
        if method not in HASH_METHOD:
            raise ValueError(f'Keep strategy [{method}] is not supported. '
                             f'Can only be one of {HASH_METHOD}.')
        self.hasher = get_hash_method(method)()

    def calculate_hash(self, sample, context=False):
        if self.image_key not in sample or not sample[self.image_key]:
            return RayBasicDeduplicator.EMPTY_HASH_VALUE

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        # compute hash
        hash_value = ''
        for key in images:
            hash_value += self.hasher.encode_image(
                image_array=np.array(images[key]))

        return hash_value
