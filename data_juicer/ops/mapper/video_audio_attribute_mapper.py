import librosa
from data_juicer.utils.constant import Fields
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.mm_utils import extract_audio_from_video
from data_juicer.my_pretrained_method.audio_code.wav2vec_age_gender import process_func,AgeGenderModel
from ..base_op import OPERATORS, Mapper
from data_juicer.utils.model_utils import get_model, prepare_model

NAME = 'video_audio_attribute_mapper'
CHECK_PKGS = [
    'transformers', 'transformers_stream_generator', 'einops', 'accelerate',
    'tiktoken'
]

with AvailabilityChecking(CHECK_PKGS, NAME):
    from data_juicer.utils.model_utils import get_model, prepare_model
    


@OPERATORS.register_module(NAME)
class VideoAudioAttributeMapper(Mapper):
    """Mapper to caption a video according to its audio streams based on
    Qwen-Audio model.
    """

    def __init__(self, 
                 hf_audio_mapper: str = None,
                 *args, **kwargs):
        """
        Initialization method.

        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only captioned sample in the
            final datasets and the original sample will be removed. It's True
            in default.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._accelerator = 'cuda'
        self._model_sampling_rate = 16000

        self._hf_summarizer = hf_audio_mapper if hf_audio_mapper else 'audeering/wav2vec2-large-robust-24-ft-age-gender'  # noqa: E501
        self.model_key = prepare_model(
            model_type='huggingface',
            pretrained_model_name_or_path=self._hf_summarizer,
        )


        

    def process(self, sample, rank=None):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # get paths of all video(s)
        loaded_video_keys = sample[self.video_key]
        audio_tag = sample['__dj__video_audio_tags__']

        Total_result = []
        # get models
        model, processor = get_model(self.model_key, rank=rank)

        for i,video in enumerate(loaded_video_keys):
            audio_tag_this = audio_tag[i]
            if not audio_tag_this == 'Speech':
                Total_result.append([])
            else:
                ys, srs, valid_indexes = extract_audio_from_video(
                    video, stream_indexes=[0])
                if len(valid_indexes) == 0:
                    # there is no valid audio streams. Skip!
                    Total_result.append([])
                    continue

                # inference
                y = ys[0]
                sr = srs[0]
                # check if it meets the sampling rate condition of the model
                if sr != self._model_sampling_rate:
                    y = librosa.resample(y,
                                        orig_sr=sr,
                                        target_sr=self._model_sampling_rate)
                    sr = self._model_sampling_rate
    
                Age_female_male_child = process_func(y, sr, processor, model, device=model.device)[0]
                Age_female_male_child_dict = {}
                Age_female_male_child_dict['Age'] = [int(Age_female_male_child[0]*100)]
                Age_female_male_child_dict['female'] = [Age_female_male_child[1]]
                Age_female_male_child_dict['male'] = [Age_female_male_child[2]]
                Age_female_male_child_dict['child'] = [Age_female_male_child[3]]
                Total_result.append([Age_female_male_child_dict])

        sample[Fields.audio_speech_attribute] = Total_result
        return sample
