import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import get_file_size, size_to_bytes

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('image_size_filter')
class ImageSizeFilter(Filter):
    """Keep data samples whose image size (in Bytes/KB/MB/...) within a
    specific range.
    """

    _batched_op = True

    def __init__(self,
                 min_size: str = '0',
                 max_size: str = '1TB',
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_size: The min image size to keep samples.  set to be "0" by
            default for no size constraint
        :param max_size: The max image size to keep samples.  set to be
            "1TB" by default, an approximate for un-limited case
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_size = size_to_bytes(min_size)
        self.max_size = size_to_bytes(max_size)
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

    def compute_stats(self, samples, context=False):
        image_list = samples[self.image_key]
        samples_stats = samples[Fields.stats]

        for i, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.image_sizes in stat:
                continue

            # there is no image in this samples
            images = image_list[i]
            if not images:
                stat[StatsKeys.image_sizes] = np.array([], dtype=np.float64)
                continue

            # for size calculation, no need to load images into memory
            stat[StatsKeys.image_sizes] = [
                get_file_size(img_path) for img_path in images
            ]

        return samples

    def process(self, samples):

        def process_single(values):
            keep_bools = np.array(
                [self.min_size <= value <= self.max_size for value in values])
            if len(keep_bools) <= 0:
                return True

            # different strategies
            if self.any:
                return keep_bools.any()
            else:
                return keep_bools.all()

        if isinstance(samples[Fields.stats], list):
            return map(
                lambda stat: process_single(stat[StatsKeys.image_sizes]),
                samples[Fields.stats])
        else:
            return process_single(samples[Fields.stats][StatsKeys.image_sizes])
