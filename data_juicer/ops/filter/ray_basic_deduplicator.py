import redis
from jsonargparse.typing import PositiveInt

from data_juicer.utils.constant import HashKeys

from ..base_op import Filter


class RayBasicDeduplicator(Filter):
    """
    A basic deduplicator to deduplicate samples at document-level using exact
    matching.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = 'EMPTY'

    def __init__(self,
                 redis_host_ip: str = 'localhost',
                 redis_host_port: PositiveInt = 6380,
                 *args,
                 **kwargs):
        """
        Initialization.
        :param redis_host_ip: the ip address of redis server
        :param redis_host_port: the port of redis server
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.redis_host_ip = redis_host_ip
        self.redis_host_port = redis_host_port

    def calculate_hash(self, sample, context=False):
        """Calculate hash value for the sample."""
        raise NotImplementedError

    def compute_stats(self, sample, context=False):
        # init redis client
        r = redis.StrictRedis(host=self.redis_host_ip,
                              port=self.redis_host_port,
                              db=0)
        # compute hash
        md5_value = self.calculate_hash(sample, context)
        # check existing
        sample[HashKeys.is_duplicate] = r.setnx(md5_value, 1)
        return sample

    def process(self, sample):
        return sample[HashKeys.is_duplicate]
