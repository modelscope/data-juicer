import numpy as np
from jsonargparse.typing import ClosedUnitInterval
from loguru import logger

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (load_data_with_context, load_image,
                                        pil_to_opencv)

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_face_ratio_filter'

with AvailabilityChecking(['dlib'], OP_NAME):
    import dlib


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFaceRatioFilter(Filter):
    """Filter to keep samples with face area ratios within a specific range.
    """

    _default_kwargs = {'upsample_num_times': 0}

    def __init__(self,
                 min_ratio: ClosedUnitInterval = 0.0,
                 max_ratio: ClosedUnitInterval = 0.4,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_ratio: Min ratio for the largest face area in an image.
        :param max_ratio: Max ratio for the largest face area in an image.
        :param any_or_all: Keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: Extra positional arguments.
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.extra_kwargs = self._default_kwargs
        for key in kwargs:
            if key in self.extra_kwargs:
                self.extra_kwargs[key] = kwargs[key]

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

        # Initialize face detector
        self.detector = dlib.get_frontal_face_detector()

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.face_ratios in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.face_ratios] = np.array(
                [], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        # detect faces
        face_detections = {}
        for key, image in images.items():
            img = pil_to_opencv(image)
            dets = self.detector(img, **self.extra_kwargs)
            face_detections[key] = [[
                max(det.left(), 0),
                max(det.top(), 0),
                min(det.right(), image.width),
                min(det.bottom(), image.height)
            ] for det in dets]
        logger.debug(f'detections: {face_detections}')

        # compute face area ratios for each image considering the largest face
        face_area_ratios = {}
        for key, dets in face_detections.items():
            image_area = images[key].width * images[key].height
            face_area_ratios[key] = max([(x2 - x1) * (y2 - y1)
                                         for x1, y1, x2, y2 in dets],
                                        default=0.0) / image_area
        logger.debug(f'ratios: {face_area_ratios}')

        sample[Fields.stats][StatsKeys.face_ratios] = [
            face_area_ratios[key] for key in loaded_image_keys
        ]
        return sample

    def process(self, sample):
        face_ratios = sample[Fields.stats][StatsKeys.face_ratios]
        if len(face_ratios) <= 0:
            return True

        keep_bools = np.array([
            self.min_ratio <= face_ratio <= self.max_ratio
            for face_ratio in face_ratios
        ])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
