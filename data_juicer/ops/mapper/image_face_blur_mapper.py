from loguru import logger

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import (load_data_with_context, load_image,
                                        pil_to_opencv)

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_face_blur_mapper'

with AvailabilityChecking(['dlib', 'Pillow'], OP_NAME):
    import dlib
    from PIL import ImageFilter


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFaceBlurMapper(Mapper):
    """Mapper to blur faces detected in images.
    """

    _default_kwargs = {'upsample_num_times': 0}

    def __init__(self,
                 blur_type: str = 'gaussian',
                 radius: float = 2,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param blur_type: Type of blur kernel, including
            ['mean', 'box', 'gaussian'].
        :param radius: Radius of blur kernel.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

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

        # Initialize face detector
        self.detector = dlib.get_frontal_face_detector()

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

        # blur face regions
        key_mapping = {}
        for key, image in images.items():
            dets = face_detections[key]
            # only blur when detected face
            if len(dets) > 0:
                blured_image = image.copy()
                for det in dets:
                    blured_roi = image.crop(det).filter(self.blur)
                    blured_image.paste(blured_roi, det)
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
