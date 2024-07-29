import av

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import (close_video, load_data_with_context,
                                        load_video, pil_to_opencv,
                                        process_each_frame)

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = 'video_face_blur_mapper'

with AvailabilityChecking(['dlib', 'Pillow'], OP_NAME):
    import dlib
    from PIL import ImageFilter


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoFaceBlurMapper(Mapper):
    """Mapper to blur faces detected in videos.
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
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.video_key]

        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)

        processed_video_keys = {}
        for video_key in loaded_video_keys:
            # skip duplicate
            if video_key in processed_video_keys:
                continue

            video = videos[video_key]
            blured_video_key = transfer_filename(video_key, OP_NAME,
                                                 **self._init_parameters)
            output_video_key = process_each_frame(video, blured_video_key,
                                                  self._blur_face)
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

    def _blur_face(self, frame):
        image = frame.to_image()
        img = pil_to_opencv(image)
        dets = self.detector(img, **self.extra_kwargs)
        if len(dets) > 0:
            for det in dets:
                x1 = max(det.left(), 0)
                y1 = max(det.top(), 0)
                x2 = min(det.right(), image.width)
                y2 = min(det.bottom(), image.height)
                blured_roi = image.crop((x1, y1, x2, y2)).filter(self.blur)
                image.paste(blured_roi, (x1, y1, x2, y2))
            frame = av.VideoFrame.from_image(image)
        return frame
