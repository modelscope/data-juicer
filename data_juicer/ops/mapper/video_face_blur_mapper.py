import os

import av

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import (close_video, detect_faces,
                                        load_data_with_context, load_video,
                                        process_each_frame)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = 'video_face_blur_mapper'

with AvailabilityChecking(['opencv-python', 'Pillow'], OP_NAME):
    import cv2
    from PIL import ImageFilter


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoFaceBlurMapper(Mapper):
    """Mapper to blur faces detected in videos.
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
                 radius: float = 2,
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
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.video_key]

        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)

        model = get_model(self.model_key)

        def _blur_func(frame):
            image = frame.to_image()
            dets = detect_faces(image, model, **self.extra_kwargs)
            if len(dets) > 0:
                for (x, y, w, h) in dets:
                    box = (x, y, x + w, y + h)
                    blured_roi = image.crop(box).filter(self.blur)
                    image.paste(blured_roi, box)
                frame = av.VideoFrame.from_image(image)
            return frame

        processed_video_keys = {}
        for video_key in loaded_video_keys:
            # skip duplicate
            if video_key in processed_video_keys:
                continue

            video = videos[video_key]
            blured_video_key = transfer_filename(video_key, OP_NAME,
                                                 **self._init_parameters)
            output_video_key = process_each_frame(video, blured_video_key,
                                                  _blur_func)
            processed_video_keys[video_key] = output_video_key

            if not context:
                close_video(video)

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(loaded_video_keys):
            if sample[Fields.source_file][i] != value:
                if processed_video_keys[value] != value:
                    sample[Fields.source_file][i] = value

        sample[self.video_key] = [
            processed_video_keys[key] for key in loaded_video_keys
        ]
        return sample
