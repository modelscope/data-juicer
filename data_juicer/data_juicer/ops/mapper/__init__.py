# yapf: disable
from . import video_audio_speech_ASR_mapper
from . import (video_captioning_face_attribute_demographic_mapper,
               video_split_by_scene_mapper,
               video_tagging_from_audio_mapper,video_audio_speech_emotion_mapper,
               video_active_speaker_mapper, video_audio_attribute_mapper,
               video_captioning_mapper_T, video_captioning_face_attribute_emotion_mapper,
               video_captioning_from_human_tracks_mapper,video_human_tracks_extraction_mapper, video_audio_speech_ASR_mapper)

from .video_captioning_from_audio_mapper import VideoCaptioningFromAudioMapper
from .video_split_by_scene_mapper import VideoSplitBySceneMapper
from .video_tagging_from_audio_mapper import VideoTaggingFromAudioMapper


# yapf: enable
