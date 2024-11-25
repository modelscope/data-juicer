import math
import os
from fractions import Fraction

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.logger_utils import HiddenPrints
from data_juicer.utils.mm_utils import close_video, load_video

from ..base_op import OPERATORS, Mapper

with HiddenPrints():
    ffmpeg = LazyLoader('ffmpeg', 'ffmpeg')

OP_NAME = 'video_resize_aspect_ratio_mapper'


def rescale(width, height, ori_ratio, min_ratio, max_ratio, strategy):

    scaled_width = width
    scaled_height = height
    ori_ratio = Fraction(ori_ratio)
    min_ratio = Fraction(min_ratio)
    max_ratio = Fraction(max_ratio)
    if ori_ratio < min_ratio:
        if strategy == 'increase':
            # increase width to meet the min ratio
            scaled_width = math.ceil(height * min_ratio)
            scaled_width += scaled_width % 2
        elif strategy == 'decrease':
            # decrease height to meet the min ratio
            scaled_height = math.floor(width / min_ratio)
            scaled_height -= scaled_height % 2

    elif ori_ratio > max_ratio:
        if strategy == 'increase':
            # increase height to meet the max ratio
            scaled_height = math.ceil(width / max_ratio)
            scaled_height += scaled_height % 2

        elif strategy == 'decrease':
            # decrease width to meet the max ratio
            scaled_width = math.floor(height * max_ratio)
            scaled_width -= scaled_width % 2

    assert Fraction(scaled_width, scaled_height) >= min_ratio
    assert Fraction(scaled_width, scaled_height) <= max_ratio

    scaled_width = max(2, scaled_width)
    scaled_height = max(2, scaled_height)

    return scaled_width, scaled_height


@OPERATORS.register_module(OP_NAME)
class VideoResizeAspectRatioMapper(Mapper):
    """Mapper to resize videos by aspect ratio.
    AspectRatio = W / H.
    """

    STRATEGY = ['decrease', 'increase']

    def __init__(
        self,
        min_ratio: str = '9/21',
        max_ratio: str = '21/9',
        strategy: str = 'increase',
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_ratio: The minimum aspect ratio to enforce videos with
            an aspect ratio below `min_ratio` will be resized to match
            this minimum ratio. The ratio should be provided as a string
            in the format "9:21" or "9/21".
        :param max_ratio: The maximum aspect ratio to enforce videos with
            an aspect ratio above `max_ratio` will be resized to match
            this maximum ratio. The ratio should be provided as a string
            in the format "21:9" or "21/9".
        :param strategy: The resizing strategy to apply when adjusting the
            video dimensions. It can be either 'decrease' to reduce the
            dimension or 'increase' to enlarge it. Accepted values are
            ['decrease', 'increase'].
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        strategy = strategy.lower()
        if strategy not in self.STRATEGY:
            raise ValueError(
                f'force_original_aspect_ratio [{strategy}] is not supported. '
                f'Can only be one of {self.STRATEGY}. ')

        self.min_ratio = Fraction(str(min_ratio).replace(':', '/'))
        self.max_ratio = Fraction(str(max_ratio).replace(':', '/'))
        self.strategy = strategy

    def process_single(self, sample):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.video_key]

        loaded_video_keys = sample[self.video_key]
        for index, video_key in enumerate(loaded_video_keys):

            container = load_video(video_key)
            video = container.streams.video[0]
            original_width = video.codec_context.width
            original_height = video.codec_context.height
            original_aspect_ratio = Fraction(original_width, original_height)
            close_video(container)

            if (original_aspect_ratio >= self.min_ratio
                    and original_aspect_ratio <= self.max_ratio):
                continue

            scaled_width, scaled_height = rescale(
                original_width,
                original_height,
                original_aspect_ratio,
                self.min_ratio,
                self.max_ratio,
                self.strategy,
            )
            resized_video_key = transfer_filename(video_key, OP_NAME,
                                                  **self._init_parameters)
            if (not os.path.exists(resized_video_key)
                    or resized_video_key not in loaded_video_keys):
                args = ['-nostdin', '-v', 'quiet', '-y']
                stream = ffmpeg.input(video_key)
                stream = stream.filter('scale',
                                       width=scaled_width,
                                       height=scaled_height)
                stream = stream.output(resized_video_key).global_args(*args)
                stream.run()
            loaded_video_keys[index] = resized_video_key

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(sample[self.video_key]):
            if sample[Fields.source_file][i] != value:
                if loaded_video_keys[i] != value:
                    sample[Fields.source_file][i] = value

        sample[self.video_key] = loaded_video_keys
        return sample
