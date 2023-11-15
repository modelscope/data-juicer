import sys

import numpy as np
from jsonargparse.typing import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES


@OPERATORS.register_module('image_shape_filter')
@LOADED_IMAGES.register_module('image_shape_filter')
class ImageShapeFilter(Filter):
    """Filter to keep samples with image shape (w, h) within specific ranges.
    """

    def __init__(self,
                 min_width: PositiveInt = 1,
                 max_width: PositiveInt = sys.maxsize,
                 min_height: PositiveInt = 1,
                 max_height: PositiveInt = sys.maxsize,
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

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.image_width in sample[Fields.stats] \
                and StatsKeys.image_height in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_width] = np.array(
                [], dtype=np.int64)
            sample[Fields.stats][StatsKeys.image_height] = np.array(
                [], dtype=np.int64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        images = {}
        for loaded_image_key in loaded_image_keys:
            if context and loaded_image_key in sample[Fields.context]:
                # load from context
                images[loaded_image_key] = sample[
                    Fields.context][loaded_image_key]
            else:
                if loaded_image_key not in images:
                    # avoid load the same images
                    image = load_image(loaded_image_key)
                    images[loaded_image_key] = image
                    if context:
                        # store the image data into context
                        sample[Fields.context][loaded_image_key] = image

        # get width and height for each image
        whs = {key: (images[key].width, images[key].height) for key in images}
        sample[Fields.stats][StatsKeys.image_width] = [
            whs[key][0] for key in loaded_image_keys
        ]
        sample[Fields.stats][StatsKeys.image_height] = [
            whs[key][1] for key in loaded_image_keys
        ]
        return sample

    def process(self, sample):
        ws = sample[Fields.stats][StatsKeys.image_width]
        hs = sample[Fields.stats][StatsKeys.image_height]
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
