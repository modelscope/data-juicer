from typing import List

import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields

from ..base_op import OPERATORS, UNFORKABLE, Filter
from ..mapper.video_tagging_from_frames_mapper import \
    VideoTaggingFromFramesMapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = 'video_tagging_from_frames_filter'

with AvailabilityChecking(
    ['torch', 'git+https://github.com/xinyu1205/recognize-anything.git'],
        OP_NAME):
    import ram  # noqa: F401
    import torch

    # avoid hanging when calling recognizeAnything in multiprocessing
    torch.set_num_threads(1)


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoTaggingFromFramesFilter(Filter):
    """Filter to keep samples whose videos contain the given tags.
    """

    _accelerator = 'cuda'

    def __init__(self,
                 tags: List[str] = ['people'],
                 contain: str = 'any',
                 frame_sampling_method: str = 'all_keyframes',
                 frame_num: PositiveInt = 3,
                 tag_field_name: str = Fields.video_frame_tags,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param tags: a tag list to shift the videos, total tags can be found
            in https://github.com/xinyu1205/recognize-anything/blob/main/ram/data/ram_tag_list.txt # noqa: E501
        :param contain: require the videos containing 'any' or 'all' tags.
            When tags equal to [], 'all' keeps all samples, 'any' keeps no
            sample.
        :param frame_sampling_method: sampling method of extracting frame
            images from the videos. Should be one of
            ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number of which depends
            on the duration of the video) and the latter one extract specified
            number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param tag_field_name: the field name to store the tags. It's
            "__dj__video_frame_tags__" in default.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if contain not in ['any', 'all']:
            raise ValueError(f'the containing type [{contain}] is not '
                             f'supported. Can only be one of ["any", "all"].')
        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method [{frame_sampling_method}] is not '
                f'supported. Can only be one of ["all_keyframes", "uniform"].')
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.tags = set([tag.lower() for tag in tags])
        self.contain_any = (contain == 'any')
        self.any = (any_or_all == 'any')
        self.tag_field_name = tag_field_name
        self.tagging_producer = VideoTaggingFromFramesMapper(
            frame_sampling_method=frame_sampling_method,
            frame_num=frame_num,
            accelerator=self.accelerator,
            tag_field_name=self.tag_field_name,
        )

    def compute_stats(self, sample, rank=None, context=False):

        sample = self.tagging_producer.process(sample, rank, context)

        return sample

    def process(self, sample, rank=None):
        video_tags = sample[self.tag_field_name]
        if len(video_tags) <= 0:
            return True

        keep_bools = []
        for words in video_tags:
            words = set([w.lower() for w in words])
            if self.contain_any:
                keep_bools.append(bool(self.tags & words))
            else:
                keep_bools.append(self.tags.issubset(words))
        keep_bools = np.array(keep_bools)

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
