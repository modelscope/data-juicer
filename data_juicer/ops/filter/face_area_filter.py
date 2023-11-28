import numpy as np
from jsonargparse.typing import ClosedUnitInterval

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_image, pil_to_opencv

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'face_area_filter'

with AvailabilityChecking(['cv2'], OP_NAME):
    import cv2  # noqa: F401

    # avoid hanging of some functions in multiprocessing
    cv2.setNumThreads(1)


class LazyCascadeClassifier:

    def __init__(self, file_path):
        self.file_path = file_path

    def __getstate__(self):
        # Only the file path is pickled
        return self.file_path

    def __setstate__(self, state):
        self.file_path = state

    def get_classifier(self):
        # Load the classifier when needed, not when pickling
        return cv2.CascadeClassifier(cv2.data.haarcascades + self.file_path)


@OPERATORS.register_module('face_area_filter')
@LOADED_IMAGES.register_module('face_area_filter')
class FaceAreaFilter(Filter):
    """Filter to keep samples with face area ratio within a specific range.
    """

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

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

        # Extract face detector arguments from kwargs
        detector_keys = [
            'scaleFactor', 'minNeighbors', 'flags', 'minSize', 'maxSize'
        ]
        self.detector_kwargs = {
            key: kwargs.pop(key)
            for key in detector_keys if key in kwargs
        }
        # Initialize face detector
        # prepare_detector()
        # self.classifier_conf = 'haarcascade_frontalface_default.xml'
        self.pickable_detector = LazyCascadeClassifier(
            'haarcascade_frontalface_default.xml')

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

        # check if faces detected already
        if StatsKeys.face_detections not in sample[Fields.stats]:
            detector = self.pickable_detector.get_classifier()
            face_detections = {}
            for key, image in images.items():
                # convert into grayscale opencv format
                opencv_img = pil_to_opencv(image, grayscale=True)
                # detect faces
                detected_faces = detector.detectMultiScale(
                    opencv_img, **self.detector_kwargs)
                # the rectangles may be partially outside the original image
                # right-closed and right-open
                face_detections[key] = [[
                    max(0, min(x, image.width - 1)),
                    max(0, min(y, image.height - 1)),
                    max(1, min(x + w, image.width)),
                    max(1, min(y + h, image.height))
                ] for (x, y, w, h) in detected_faces]

            sample[Fields.stats][StatsKeys.face_detections] = [
                face_detections[key] for key in loaded_image_keys
            ]

        sample[Fields.stats][StatsKeys.face_ratios] = [
            max([((x2 - x1) * (y2 - y1)) /
                 (images[key].width * images[key].height)
                 for x1, y1, x2, y2 in dets],
                default=0) for key, dets in zip(
                    loaded_image_keys, sample[Fields.stats][
                        StatsKeys.face_detections])
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
