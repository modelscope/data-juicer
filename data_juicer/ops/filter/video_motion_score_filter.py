import sys
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import numpy as np
from pydantic import PositiveFloat, PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, UNFORKABLE, Filter

OP_NAME = 'video_motion_score_filter'

with AvailabilityChecking(['opencv-python'], OP_NAME):
    import cv2


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoMotionScoreFilter(Filter):
    """Filter to keep samples with video motion scores within a specific range. The
    Farneback's algorith from OpenCV is used to compute dense optical flow.
    """

    _default_kwargs = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }

    def __init__(self,
                 min_score: float = 0.25,
                 max_score: float = sys.float_info.max,
                 sampling_fps: PositiveFloat = 2,
                 size: Union[PositiveInt, Tuple[PositiveInt],
                             Tuple[PositiveInt, PositiveInt], None] = None,
                 max_size: Optional[PositiveInt] = None,
                 relative: bool = False,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_score: The minimum motion score to keep samples.
        :param max_score: The maximum motion score to keep samples.
        :param sampling_fps: The sampling rate in frames_per_second for
            optical flow calculations.
        :param size: Resize frames before computing optical flow. If size is a
            sequence like (h, w), frame size will be matched to this. If size
            is an int, smaller edge of frames will be matched to this number.
            i.e, if height > width, then frame will be rescaled to (size *
            height / width, size). Default `None` to keep the original size.
        :param max_size: The maximum allowed for the longer edge of resized
            frames. If the longer edge of frames is greater than max_size after
            being resized according to size, size will be overruled so that the
            longer edge is equal to max_size. As a result, the smaller edge may
            be shorter than size. This is only supported if size is an int.
        :param relative: If `True`, the optical flow magnitude is normalized to
            a [0, 1] range, relative to the frame's diagonal length.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        self.sampling_fps = sampling_fps

        if isinstance(size, (list, tuple)):
            if len(size) not in [1, 2]:
                raise ValueError(
                    f'Size must be an int or a 1 or 2 element tuple/list,'
                    f'not a {len(size)} element tuple/list.')
        if isinstance(size, int):
            size = (size, )
        self.size = size
        self.max_size = max_size
        self.relative = relative

        self.extra_kwargs = self._default_kwargs
        for key in kwargs:
            if key in self.extra_kwargs:
                self.extra_kwargs[key] = kwargs[key]

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.video_motion_score in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_motion_score] = np.array(
                [], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        unique_motion_scores = {}
        for video_key in loaded_video_keys:
            # skip duplicate videos
            if video_key in unique_motion_scores:
                continue

            video_motion_scores = []
            with VideoCapture(video_key) as cap:
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    sampling_fps = min(self.sampling_fps, fps)
                    sampling_step = round(fps / sampling_fps)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # at least two frames for computing optical flow
                    sampling_step = max(min(sampling_step, total_frames - 1),
                                        1)

                prev_frame = None
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        # If the frame can't be read, it could be due to
                        # a corrupt frame or reaching the end of the video.
                        break

                    height, width, _ = frame.shape
                    new_size = _compute_resized_output_size(
                        (height, width), self.size, self.max_size)
                    if new_size != (height, width):
                        frame = cv2.resize(frame,
                                           new_size,
                                           interpolation=cv2.INTER_AREA)

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_frame is None:
                        prev_frame = gray_frame
                        continue

                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, gray_frame, None, **self.extra_kwargs)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    frame_motion_score = np.mean(mag)
                    if self.relative:
                        frame_motion_score /= np.hypot(*flow.shape[:2])
                    video_motion_scores.append(frame_motion_score)
                    prev_frame = gray_frame

                    # quickly skip frames
                    frame_count += sampling_step
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            # may due to frame corruption
            if not video_motion_scores:
                unique_motion_scores[video_key] = -1
            else:
                unique_motion_scores[video_key] = np.mean(video_motion_scores
                                                          or [-1])

        sample[Fields.stats][StatsKeys.video_motion_score] = [
            unique_motion_scores[key] for key in loaded_video_keys
        ]
        return sample

    def process(self, sample):
        video_motion_scores = sample[Fields.stats][
            StatsKeys.video_motion_score]

        keep_bools = np.array([
            self.min_score <= motion_score <= self.max_score
            for motion_score in video_motion_scores
        ])
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()


def _compute_resized_output_size(
    frame_size: Tuple[int, int],
    size: Union[Tuple[PositiveInt], Tuple[PositiveInt, PositiveInt]],
    max_size: Optional[int] = None,
) -> Tuple[int, int]:
    h, w = frame_size
    short, long = (w, h) if w <= h else (h, w)

    if size is None:  # no change
        new_short, new_long = short, long
    elif len(size) == 1:  # specified size only for the smallest edge
        new_short = size[0]
        new_long = int(new_short * long / short)
    else:  # specified both h and w
        new_short, new_long = min(size), max(size)

    if max_size is not None and new_long > max_size:
        new_short = int(max_size * new_short / new_long)
        new_long = max_size

    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return new_h, new_w
