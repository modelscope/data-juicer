import ray

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import Filter

redis = LazyLoader('redis', 'redis')

MERSENNE_PRIME = (1 << 61) - 1


@ray.remote(scheduling_strategy='SPREAD')
class DedupSet:
    def __init__(self):
        self.hash_record = set()

    def setnx(self, key):
        if key not in self.hash_record:
            self.hash_record.add(key)
            return True
        else:
            return False


class RayBasicDeduplicator(Filter):
    """
    A basic exact matching deduplicator for RAY.
    Although its functionality is deduplication,
    it is implemented as Filter sub-class.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = 'EMPTY'

    def __init__(self,
                 backend: str = 'ray_actor',
                 redis_address: str = 'redis://localhost:6379',
                 *args,
                 **kwargs):
        """
        Initialization.
        :param backend: the backend for dedup, either 'ray_actor' or 'redis'
        :param redis_address: the address of redis server
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.redis_address = redis_address
        self.backend = backend
        if backend == 'ray_actor':
            self.dedup_set_num = int(ray.cluster_resources().get('CPU') / 2)
            self.dedup_sets = [
                DedupSet.remote() for _ in range(self.dedup_set_num)
            ]
        elif backend == 'redis':
            # TODO: add a barrier to ensure that flushdb is performed before
            # the operator is called
            r = redis.from_url(url=self.redis_address)
            r.flushdb(0)
        else:
            raise ValueError(f'Unknown backend: {backend}')

    def calculate_hash(self, sample, context=False):
        """Calculate hash value for the sample."""
        raise NotImplementedError

    def compute_stats_single(self, sample, context=False):
        if self.backend == 'ray_actor':
            # compute hash
            md5_value = self.calculate_hash(sample, context)
            dedup_set_id = int.from_bytes(
                md5_value.encode(),
                byteorder='little'
            ) % MERSENNE_PRIME % self.dedup_set_num
            # check existing
            sample[HashKeys.is_unique] = \
                ray.get(self.dedup_sets[dedup_set_id].setnx.remote(md5_value))
            return sample
        else:  # redis
            # init redis client
            r = redis.from_url(url=self.redis_address)
            # compute hash
            md5_value = self.calculate_hash(sample, context)
            # check existing
            sample[HashKeys.is_unique] = r.setnx(md5_value, 1)
            return sample

    def process_single(self, sample):
        return sample[HashKeys.is_unique]
