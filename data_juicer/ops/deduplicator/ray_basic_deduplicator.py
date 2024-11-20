from pydantic import PositiveInt

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import Filter

redis = LazyLoader('redis', 'redis')


class RayBasicDeduplicator(Filter):
    """
    A basic exact matching deduplicator for RAY.
    Although its functionality is deduplication,
    it is implemented as Filter sub-class.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = 'EMPTY'

    def __init__(self,
                 redis_address: str = 'redis://localhost:6379',
                 *args,
                 **kwargs):
        """
        Initialization.
        :param redis_address: the address of redis server
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.redis_address = redis_address
        # TODO: add a barrier to ensure that flushdb is performed before
        # the operator is called
        r = redis.from_url(url=redis_address)
        r.flushdb(0)

    def calculate_hash(self, sample, context=False):
        """Calculate hash value for the sample."""
        raise NotImplementedError

    def compute_stats_single(self, sample, context=False):
        # init redis client
        r = redis.from_url(url=self.redis_address)
        # compute hash
        md5_value = self.calculate_hash(sample, context)
        # check existing
        sample[HashKeys.is_duplicate] = r.setnx(md5_value, 1)
        return sample

    def process_single(self, sample):
        return sample[HashKeys.is_duplicate]
