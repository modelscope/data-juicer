import sys

import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES


@OPERATORS.register_module('image_shape_filter')
@LOADED_IMAGES.register_module('image_shape_filter')
class ImageShapeFilter(Filter):
    """Filter to keep samples with image shape (w, h) within specific ranges.
    """

    _batched_op = True

    def __init__(self,
                 min_width: int = 1,
                 max_width: int = sys.maxsize,
                 min_height: int = 1,
                 max_height: int = sys.maxsize,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_width: The min width to keep samples.
        :param max_width: The max width to keep samples.
        :param min_height: The min height to keep samples.
        :param max_height: The max height to keep samples.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

    def compute_stats(self, samples, context=False):
        image_list = samples[self.image_key]
        samples_stats = samples[Fields.stats]

        for i, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.image_width in stat \
                    and StatsKeys.image_height in stat:
                continue

            # there is no image in this samples
            loaded_image_keys = image_list[i]
            if not loaded_image_keys:
                stat[StatsKeys.image_width] = np.array([], dtype=np.int64)
                stat[StatsKeys.image_height] = np.array([], dtype=np.int64)
                continue

            # load images
            samples, images = load_data_with_context(samples, context,
                                                     loaded_image_keys,
                                                     load_image)

            # get width and height for each image
            whs = {
                key: (images[key].width, images[key].height)
                for key in images
            }
            stat[StatsKeys.image_width] = [
                whs[key][0] for key in loaded_image_keys
            ]
            stat[StatsKeys.image_height] = [
                whs[key][1] for key in loaded_image_keys
            ]

        return samples

    def process(self, samples):

        def process_single(ws, hs):
            if len(ws) <= 0:
                return True
            keep_bools = np.array([
                self.min_width <= w <= self.max_width
                and self.min_height <= h <= self.max_height
                for w, h in zip(ws, hs)
            ])

            # different strategies
            if self.any:
                return keep_bools.any()
            else:
                return keep_bools.all()

        if isinstance(samples[Fields.stats], list):
            return map(
                lambda stat: process_single(stat[StatsKeys.image_width], stat[
                    StatsKeys.image_height]), samples[Fields.stats])
        else:
            return process_single(
                samples[Fields.stats][StatsKeys.image_width],
                samples[Fields.stats][StatsKeys.image_height])
