import os

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

with AvailabilityChecking(['opencv-python'], OP_NAME):
    import cv2

OPENCV_DETECTOR = None


def prepare_detector(cv_classifier):
    global OPENCV_DETECTOR
    if OPENCV_DETECTOR is None:
        OPENCV_DETECTOR = cv2.CascadeClassifier(cv_classifier)


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFaceRatioFilter(Filter):
    """Filter to keep samples with face area ratios within a specific range.
    """

    _default_kwargs = {
        'scaleFactor': 1.1,
        'minNeighbors': 3,
        'minSize': None,
        'maxSize': None,
    }

    def __init__(self,
                 cv_classifier='',
                 min_ratio: ClosedUnitInterval = 0.0,
                 max_ratio: ClosedUnitInterval = 0.4,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param cv_classifier: OpenCV classifier path for face detection.
            By default, we will use 'haarcascade_frontalface_alt.xml'.
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

        if cv_classifier == '':
            cv_classifier = os.path.join(cv2.data.haarcascades,
                                         'haarcascade_frontalface_alt.xml')
        self.cv_classifier = cv_classifier

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
        prepare_detector(self.cv_classifier)

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
        prepare_detector(self.cv_classifier)
        face_detections = {}
        for key, image in images.items():
            img = pil_to_opencv(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = OPENCV_DETECTOR.detectMultiScale(gray, **self.extra_kwargs)
            rectified_dets = []
            for (x, y, w, h) in dets:
                x = max(x, 0)
                y = max(y, 0)
                w = min(w, image.width - x)
                h = min(h, image.height - y)
                rectified_dets.append([x, y, w, h])
            face_detections[key] = rectified_dets
        logger.debug(f'detections: {face_detections}')

        # compute face area ratios for each image considering the largest face
        face_area_ratios = {}
        for key, dets in face_detections.items():
            image_area = images[key].width * images[key].height
            face_area_ratios[key] = max([w * h for _, _, w, h in dets],
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
