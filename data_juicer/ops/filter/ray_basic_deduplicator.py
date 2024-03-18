import redis

from data_juicer.utils.constant import HashKeys
from ..base_op import Filter


class RayBasicDeduplicator(Filter):
    """
    A basic deduplicator to deduplicate samples at document-level using exact matching.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = "EMPTY"

    def __init__(self, *args, **kwargs):
        """
        Initialization.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def calculate_hash(self, sample, context=False):
        """Calculate hash value for the sample."""
        raise NotImplementedError

    def compute_stats(self, sample, context=False):
        # init redis client
        r = redis.StrictRedis(host="localhost", port=6379, db=0)
        # compute hash
        md5_value = self.calculate_hash(sample, context)
        # check existing
        sample[HashKeys.is_duplicate] = r.setnx(md5_value, 1)
        return sample

    def process(self, sample):
        return sample[HashKeys.is_duplicate]
