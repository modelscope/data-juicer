import av
import numpy as np
from jsonargparse.typing import ClosedUnitInterval
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (load_data_with_context, load_video,
                                        pil_to_opencv, pil_to_numpy, process_each_frame)

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_VIDEOS

OP_NAME = 'video_face_ratio_filter'

with AvailabilityChecking(['dlib', 'Pillow'], OP_NAME):
    import dlib
    from PIL import ImageFilter

@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoFaceRatioFilter(Filter):
    """Keep data samples whose videos' durations are within a specified range.
    """

    def __init__(self,
                 threshold: ClosedUnitInterval = 0.8,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.threshold = threshold

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

        # Initialize face detector
        self.detector = dlib.get_frontal_face_detector()

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.video_face_exist in sample[Fields.stats]:
            return sample
        
        # load videos
        loaded_video_keys = sample[self.video_key]
        video_faces_ratio = {}

        for video_key in loaded_video_keys:
            container = av.open(video_key)

            # 获取视频流信息
            video_stream = next(s for s in container.streams if s.type == 'video')

            # 遍历视频帧，检测人脸
            total_frames = 0
            frames_with_face = 0

            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    total_frames += 1
                    img = frame.to_image()
                    image = pil_to_numpy(img)
                    faces = self.detector(image)
                    if faces:
                        frames_with_face += 1
            
            # 计算人脸帧数占比
            if total_frames > 0:
                face_ratio = frames_with_face / total_frames
            else:
                face_ratio = 0.0

            video_faces_ratio[video_key] = face_ratio

        # get video faces ratio
        sample[Fields.stats][StatsKeys.video_face_exist] = [
            video_faces_ratio[video_key] for video_key in sample[self.video_key]
        ]

        return sample

    def process(self, sample):
        video_faces_ratio = sample[Fields.stats][StatsKeys.video_face_exist]
        keep_bools = np.array([
            duration >= self.threshold
            for duration in video_faces_ratio
        ])
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()

