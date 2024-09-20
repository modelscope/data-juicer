import math
import re
from itertools import chain

from pydantic import NonNegativeFloat, NonNegativeInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import (add_suffix_to_filename,
                                          transfer_filename)
from data_juicer.utils.mm_utils import SpecialTokens

from ..base_op import OPERATORS, Mapper

OP_NAME = 'video_split_by_scene_mapper'

with AvailabilityChecking(['scenedetect[opencv]'], OP_NAME):
    import scenedetect.detectors
    from scenedetect import detect, split_video_ffmpeg


def replace_func(match, scene_counts_iter):
    try:
        count = next(scene_counts_iter)
        return SpecialTokens.video * count
    except StopIteration:
        return match.group(0)


@OPERATORS.register_module(OP_NAME)
class VideoSplitBySceneMapper(Mapper):
    """Mapper to cut videos into scene clips.
    """

    # Define shared detector keys and their properties
    avaliable_detectors = {
        'ContentDetector': ['weights', 'luma_only', 'kernel_size'],
        'AdaptiveDetector': [
            'window_width', 'min_content_val', 'weights', 'luma_only',
            'kernel_size', 'video_manager', 'min_delta_hsv'
        ],
        'ThresholdDetector':
        ['fade_bias', 'add_final_scene', 'method', 'block_size']
    }

    def __init__(self,
                 detector: str = 'ContentDetector',
                 threshold: NonNegativeFloat = 27.0,
                 min_scene_len: NonNegativeInt = 15,
                 show_progress: bool = False,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param detector: Algorithm from `scenedetect.detectors`. Should be one
            of ['ContentDetector', 'ThresholdDetector', 'AdaptiveDetector`].
        :param threshold: Threshold passed to the detector.
        :param min_scene_len: Minimum length of any scene.
        :param show_progress: Whether to show progress from scenedetect.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        if detector not in self.avaliable_detectors:
            raise ValueError(
                f'Scene detector {detector} is not supported. '
                f'Can only be one of {list(self.avaliable_detectors.keys())}')

        self.detector = detector
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.show_progress = show_progress

        # prepare detector args
        avaliable_kwargs = self.avaliable_detectors[self.detector]
        self.detector_class = getattr(scenedetect.detectors, self.detector)
        self.detector_kwargs = {
            key: kwargs[key]
            for key in avaliable_kwargs if key in kwargs
        }

    def process(self, sample, context=False):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        output_video_keys = {}
        scene_counts = {}

        for video_key in loaded_video_keys:

            # skip duplicate
            if video_key in output_video_keys:
                continue

            redirected_video_key = transfer_filename(video_key, OP_NAME,
                                                     **self._init_parameters)
            output_template = add_suffix_to_filename(redirected_video_key,
                                                     '_$SCENE_NUMBER')

            # detect scenes
            detector = self.detector_class(self.threshold, self.min_scene_len,
                                           **self.detector_kwargs)
            scene_list = detect(video_key,
                                detector,
                                show_progress=self.show_progress,
                                start_in_scene=True)
            scene_counts[video_key] = len(scene_list)

            if len(scene_list) > 1:
                # sync with split_video_ffmpeg internal
                scene_num_format = f'%0{max(3, math.floor(math.log(len(scene_list), 10)) + 1)}d'  # noqa: E501
                output_video_keys[video_key] = [
                    output_template.replace('$SCENE_NUMBER',
                                            scene_num_format % (i + 1))
                    for i in range(len(scene_list))
                ]
                # split video into clips
                split_video_ffmpeg(input_video_path=video_key,
                                   scene_list=scene_list,
                                   output_file_template=output_template,
                                   show_progress=self.show_progress)
            else:
                output_video_keys[video_key] = [video_key]

        # replace splited video tokens
        if self.text_key in sample:
            scene_counts_iter = iter(
                [scene_counts[key] for key in loaded_video_keys])
            updated_text = re.sub(
                re.escape(SpecialTokens.video),
                lambda match: replace_func(match, scene_counts_iter),
                sample[self.text_key])
            sample[self.text_key] = updated_text

        # when the file is modified, its source file needs to be updated.
        sample[Fields.source_file] = []
        for value in loaded_video_keys:
            sample[Fields.source_file].extend([value] *
                                              len(output_video_keys[value]))

        sample[self.video_key] = list(
            chain.from_iterable(
                [output_video_keys[key] for key in loaded_video_keys]))
        return sample
