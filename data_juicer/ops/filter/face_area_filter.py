import numpy as np
from jsonargparse.typing import ClosedUnitInterval

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_image, pil_to_opencv

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'face_area_filter'

with AvailabilityChecking(['dlib'], OP_NAME):
    import dlib


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
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

        # Extract face detector arguments from kwargs
        detector_keys = ['upsample_num_times']
        self.detector_kwargs = {
            key: kwargs.pop(key)
            for key in detector_keys if key in kwargs
        }

        super().__init__(*args, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

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

        # there is no image in this sample, still default ratio 0.0
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.face_ratios] = [0.0]
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
            face_detections = {}
            for key, image in images.items():
                img = pil_to_opencv(image)
                dets = self.detector(img, **self.detector_kwargs)
                dets_formatted = [[
                    det.left(),
                    det.top(),
                    det.width(),
                    det.height()
                ] for det in dets] if dets else [[0, 0, 0, 0]]
                face_detections[key] = dets_formatted
            sample[Fields.stats][StatsKeys.face_detections] = [
                face_detections[key] for key in loaded_image_keys
            ]

        max_face_ratios = []
        for key, dets in zip(loaded_image_keys,
                             sample[Fields.stats][StatsKeys.face_detections]):
            img_area = images[key].width * images[key].height
            # Calculate the max face ratio for the current image
            max_face_ratios.append(
                max([w * h / img_area for _, _, w, h in dets]))
        sample[Fields.stats][StatsKeys.face_ratios] = max_face_ratios

        return sample

    def process(self, sample):
        if self.image_key not in sample or not sample[self.image_key]:
            return True

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
