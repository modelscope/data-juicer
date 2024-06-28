import sys
from contextlib import contextmanager
from typing import Optional

import numpy as np
from jsonargparse.typing import PositiveInt
from loguru import logger

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter

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
                 sampling_fps: float = 2,
                 target_size: Optional[PositiveInt] = None,
                 normalize: bool = False,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param min_score: The minimum motion score to keep samples.
        :param max_score: The maximum motion score to keep samples.
        :param sampling_fps: The samplig rate in frames per second for
            optical flow calculations.
        :param target_size: Resize frames along the smaller edge before
            computing optical flow. Set to `None` to keep the original size.
        :param normalize: If `True`, normalizes motion scores to a [0, 1]
            range relative to the frame's diagonal length.
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
        self.target_size = target_size
        self.normalize = normalize

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
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    logger.warning(f'Failed to get FPS from video; skipping {video_key}.')
                    break

                sampling_fps = self.sampling_fps
                if self.sampling_fps > fps:
                    sampling_fps = fps
                    logger.warning(
                        f'sampling_fps {self.sampling_fps} is higher than video fps {fps};'
                        'setting sampling_fps to video fp.')

                sampling_stride = int(fps / sampling_fps)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    logger.warning(f'Video has no frames; skipping {video_key}.')
                    break
                elif total_frames == 1:
                    video_motion_scores.append(0)
                    break
                # if cannot get the second frame, use the last one
                sampling_stride = min(sampling_stride, total_frames - 1)

                prev_frame = None
                frame_count = -1
                while cap.isOpened():
                    ret, frame = cap.read()
                    frame_count += 1

                    if not ret:
                        # If the frame can't be read, it could be due to
                        # a corrupt frame or reaching the end of the video.
                        break

                    # skip intermediate frames
                    if frame_count % sampling_stride != 0:
                        continue

                    if self.target_size is not None:
                        height, width, _ = frame.shape
                        scale_ratio = self.target_size / min(width, height)
                        new_width = int(width * scale_ratio)
                        new_height = int(height * scale_ratio)
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_frame is None:
                        prev_frame = gray_frame
                        continue

                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, gray_frame, None, **self.extra_kwargs)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    frame_motion_score = np.mean(mag)
                    if self.normalize:
                        frame_motion_score /= np.hypot(*flow.shape[:2])
                    video_motion_scores.append(frame_motion_score)
                    prev_frame = gray_frame

            # set to -1 upon loop break
            unique_motion_scores[video_key] = np.mean(video_motion_scores or [-1])

        sample[Fields.stats][StatsKeys.video_motion_score] = [
            unique_motion_scores[key] for key in loaded_video_keys
        ]
        print(sample)
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
