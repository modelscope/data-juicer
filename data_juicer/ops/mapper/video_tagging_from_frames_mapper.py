from collections import Counter

from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import (close_video, extract_key_frames,
                                        extract_video_frames_uniformly,
                                        load_data_with_context, load_video)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = 'video_tagging_from_frames_mapper'

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
class VideoTaggingFromFramesMapper(Mapper):
    """Mapper to generate video tags from frames extract by video.
    """

    _accelerator = 'cuda'

    def __init__(self,
                 frame_sampling_method: str = 'all_keyframes',
                 frame_num: PositiveInt = 3,
                 *args,
                 **kwargs):
        """
        Initialization method.

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
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method [{frame_sampling_method}] is not '
                f'supported. Can only be one of ["all_keyframes", "uniform"].')
        self.model_key = prepare_model(
            model_type='recognizeAnything',
            pretrained_model_name_or_path='ram_plus_swin_large_14m.pth',
            input_size=384)
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        from ram import get_transform
        self.transform = get_transform(image_size=384)

    def process(self, sample, rank=None, context=False):
        # check if it's generated already
        if Fields.video_frame_tags in sample:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.video_frame_tags] = []
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)

        model = get_model(self.model_key, rank, self.use_cuda())
        video_tags = []
        for _, value in enumerate(loaded_video_keys):
            video = videos[value]

            # extract frame images
            if self.frame_sampling_method == 'all_keyframes':
                frames = extract_key_frames(video)
            elif self.frame_sampling_method == 'uniform':
                frames = extract_video_frames_uniformly(video, self.frame_num)
            else:
                video_tags.append([])
                frames = []

            frame_tensor = torch.stack([
                self.transform(frame.to_image()) for frame in frames
            ]).to(next(model.parameters()).device)
            with torch.no_grad():
                tags, _ = model.generate_tag(frame_tensor)

            words = [word.strip() for tag in tags for word in tag.split('|')]
            word_count = Counter(words)
            sorted_word_list = [item for item, _ in word_count.most_common()]
            video_tags.append(sorted_word_list)

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])

        sample[Fields.video_frame_tags] = video_tags
        return sample
