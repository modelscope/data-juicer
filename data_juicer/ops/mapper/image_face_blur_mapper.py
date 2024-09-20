import os

from loguru import logger
from pydantic import NonNegativeFloat

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import (detect_faces, load_data_with_context,
                                        load_image)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_face_blur_mapper'

with AvailabilityChecking(['opencv-python', 'Pillow'], OP_NAME):
    import cv2
    from PIL import ImageFilter


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFaceBlurMapper(Mapper):
    """Mapper to blur faces detected in images.
    """

    _default_kwargs = {
        'scaleFactor': 1.1,
        'minNeighbors': 3,
        'minSize': None,
        'maxSize': None,
    }

    def __init__(self,
                 cv_classifier: str = '',
                 blur_type: str = 'gaussian',
                 radius: NonNegativeFloat = 2,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param cv_classifier: OpenCV classifier path for face detection.
            By default, we will use 'haarcascade_frontalface_alt.xml'.
        :param blur_type: Type of blur kernel, including
            ['mean', 'box', 'gaussian'].
        :param radius: Radius of blur kernel.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        if cv_classifier == '':
            cv_classifier = os.path.join(cv2.data.haarcascades,
                                         'haarcascade_frontalface_alt.xml')
        if blur_type not in ['mean', 'box', 'gaussian']:
            raise ValueError(
                f'Blur_type [{blur_type}] is not supported. '
                f'Can only be one of ["mean", "box", "gaussian"]. ')
        if radius < 0:
            raise ValueError('Radius must be >= 0. ')

        if blur_type == 'mean':
            self.blur = ImageFilter.BLUR
        elif blur_type == 'box':
            self.blur = ImageFilter.BoxBlur(radius)
        else:
            self.blur = ImageFilter.GaussianBlur(radius)

        self.blur_type = blur_type
        self.radius = radius

        self.extra_kwargs = self._default_kwargs
        for key in kwargs:
            if key in self.extra_kwargs:
                self.extra_kwargs[key] = kwargs[key]

        self.model_key = prepare_model(model_type='opencv_classifier',
                                       model_path=cv_classifier)

    def process(self, sample, context=False):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.image_key]

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        model = get_model(self.model_key)

        # detect faces
        face_detections = {}
        for key, image in images.items():
            face_detections[key] = detect_faces(image, model,
                                                **self.extra_kwargs)
        logger.debug(f'detections: {face_detections}')

        # blur face regions
        key_mapping = {}
        for key, image in images.items():
            dets = face_detections[key]
            # only blur when detected face
            if len(dets) > 0:
                blured_image = image.copy()
                for (x, y, w, h) in dets:
                    box = (x, y, x + w, y + h)
                    blured_roi = image.crop(box).filter(self.blur)
                    blured_image.paste(blured_roi, box)
                blured_image_key = transfer_filename(key, OP_NAME,
                                                     **self._init_parameters)
                blured_image.save(blured_image_key)
                key_mapping[key] = blured_image_key
                if context:
                    sample[Fields.context][blured_image_key] = blured_image
            else:
                key_mapping[key] = key

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(loaded_image_keys):
            if sample[Fields.source_file][i] != value:
                if key_mapping[value] != value:
                    sample[Fields.source_file][i] = value

        sample[self.image_key] = [
            key_mapping[key] for key in loaded_image_keys
        ]
        return sample
