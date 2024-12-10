import json
import os
import os.path as osp

from pydantic import PositiveInt

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import create_directory_if_not_exists
from data_juicer.utils.mm_utils import (SpecialTokens, close_video,
                                        extract_key_frames,
                                        extract_video_frames_uniformly,
                                        load_data_with_context, load_video)

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = 'video_extract_frames_mapper'


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoExtractFramesMapper(Mapper):
    """Mapper to extract frames from video files according to specified methods.
    Extracted Frames Data Format:
        The data format for the extracted frames is a dictionary mapping
        video keys to lists of file paths where the extracted frames are saved.
        The dictionary follows the structure:
        {
            "video_key_1": [
                "/${frame_dir}/video_key_1_filename/frame_1.jpg",
                "/${frame_dir}/video_key_1_filename/frame_2.jpg",
                ...],
            "video_key_2": [
                "/${frame_dir}/video_key_2_filename/frame_1.jpg",
                "/${frame_dir}/video_key_2_filename/frame_2.jpg",
                ...],
            ...
        }
    """

    _batched_op = True

    def __init__(
        self,
        frame_sampling_method: str = 'all_keyframes',
        frame_num: PositiveInt = 3,
        frame_dir: str = None,
        frame_key=Fields.video_frames,
        *args,
        **kwargs,
    ):
        """
        Initialization method.
        :param frame_sampling_method: sampling method of extracting frame
            videos from the videos. Should be one of
            ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number
            of which depends on the duration of the video) and the latter
            one extract specified number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param frame_dir: Output directory to save extracted frames.
            If None, a default directory based on the video file path is used.
        :param frame_key: The name of field to save generated frames info.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method '
                f'[{frame_sampling_method}] is not supported. '
                f'Can only be one of ["all_keyframes", "uniform"].')

        self.frame_dir = frame_dir
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        self.frame_key = frame_key
        self.frame_fname_template = 'frame_{}.jpg'

    def _get_default_frame_dir(self, original_filepath):
        original_dir = os.path.dirname(original_filepath)
        dir_token = f'/{Fields.multimodal_data_output_dir}/'
        if dir_token in original_dir:
            original_dir = original_dir.split(dir_token)[0]
        new_dir = os.path.join(
            original_dir, f'{Fields.multimodal_data_output_dir}/{OP_NAME}')
        create_directory_if_not_exists(new_dir)
        return osp.join(new_dir,
                        osp.splitext(osp.basename(original_filepath))[0])

    def process_single(self, sample, context=False):
        # check if it's generated already
        if self.frame_key in sample:
            return sample

        # there is no videos in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)
        video_to_frames = {}
        text = sample[self.text_key]
        offset = 0

        for chunk in text.split(SpecialTokens.eoc):
            video_count = chunk.count(SpecialTokens.video)
            # no video or no text
            if video_count == 0 or len(chunk) == 0:
                continue
            else:
                for video_key in loaded_video_keys[offset:offset +
                                                   video_count]:
                    video = videos[video_key]
                    # extract frame videos
                    if self.frame_sampling_method == 'all_keyframes':
                        frames = extract_key_frames(video)
                    elif self.frame_sampling_method == 'uniform':
                        frames = extract_video_frames_uniformly(
                            video, self.frame_num)
                    else:
                        raise ValueError(f'Not support sampling method \
                            `{self.frame_sampling_method}`.')
                    frames = [frame.to_image() for frame in frames]

                    if self.frame_dir:
                        frame_dir = osp.join(
                            self.frame_dir,
                            osp.splitext(osp.basename(video_key))[0])
                    else:
                        # video path as frames directory
                        frame_dir = self._get_default_frame_dir(video_key)
                    os.makedirs(frame_dir, exist_ok=True)

                    video_to_frames[video_key] = []
                    for i, frame in enumerate(frames):
                        frame_path = osp.join(
                            frame_dir, self.frame_fname_template.format(i))
                        if not os.path.exists(frame_path):
                            frame.save(frame_path)

                        video_to_frames[video_key].append(frame_path)

                offset += video_count

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])

        sample[self.frame_key] = json.dumps(video_to_frames)
        # sample[self.frame_key] = video_to_frames

        return sample