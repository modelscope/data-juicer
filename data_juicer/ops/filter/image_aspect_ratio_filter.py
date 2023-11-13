import numpy as np
from jsonargparse.typing import PositiveFloat

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES


@OPERATORS.register_module('image_aspect_ratio_filter')
@LOADED_IMAGES.register_module('image_aspect_ratio_filter')
class ImageAspectRatioFilter(Filter):
    """Filter to keep samples with image aspect ratio within a specific range.
    AspectRatio = W / H.
    """

    def __init__(self,
                 min_ratio: PositiveFloat = 0.333,
                 max_ratio: PositiveFloat = 3.0,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_ratio: The min aspect ratio to keep samples.
        :param max_ratio: The max aspect ratio to keep samples.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.aspect_ratios in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.aspect_ratios] = np.array(
                [], dtype=np.float64)
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

        # compute aspect ratios for each image with W/H
        aspect_ratios = {
            key: (images[key].width / images[key].height)
            for key in images
        }
        sample[Fields.stats][StatsKeys.aspect_ratios] = [
            aspect_ratios[key] for key in loaded_image_keys
        ]
        return sample

    def process(self, sample):
        aspect_ratios = sample[Fields.stats][StatsKeys.aspect_ratios]
        keep_bools = np.array([
            self.min_ratio <= aspect_ratio <= self.max_ratio
            for aspect_ratio in aspect_ratios
        ])
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
