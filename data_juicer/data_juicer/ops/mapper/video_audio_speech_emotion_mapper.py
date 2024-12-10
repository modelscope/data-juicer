import librosa

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import extract_audio_from_video
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
import gc

# import sys
# sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/SenseVoice')
from data_juicer.my_pretrained_method.SenseVoice.model import SenseVoiceSmall


OP_NAME = 'video_audio_speech_emotion_mapper'

with AvailabilityChecking(['torch', 'transformers', 'torchaudio'], OP_NAME):
    import torch
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
class VideoAudioSpeechEmotionMapper(Mapper):
    """Mapper to generate video tags from audio streams extracted by video
    using the Audio Spectrogram Transformer.
    """

    def __init__(self,
                 model_dir_emo='/mnt1/daoyuan_mm/SenseVoiceSmall',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._batched_op = True
        self._accelerator = 'cuda'
        self._model_sampling_rate = 16000
        self.model_dir_emo = model_dir_emo

        self.model_key = prepare_model(
            model_type='SenseVoiceSmall',
            pretrained_model_name_or_path=self.model_dir_emo,
        )

    def process(self, sample, rank=None):
        # check if it's generated already
        if Fields.speech_emotion in sample:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.video_audio_tags] = []
            return sample

        # load video paths
        loaded_video_keys = sample[self.video_key][0]
        audio_tags = sample[Fields.video_audio_tags][0]

        # model, feature_extractor = get_model(self.model_key, rank=rank)
        video_audio_tags = []
        for id,video_path in enumerate(loaded_video_keys):
            if audio_tags[id] == 'Speech':
                # only extract audio data and sr for index 0 for now
                ys, srs, valid_indexes = extract_audio_from_video(
                    video_path, stream_indexes=[0])
                if len(valid_indexes) == 0:
                    # there is no valid audio streams. Skip!
                    video_audio_tags.append(self._no_audio_label)
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
                
                ASR_Emo_model, kwargs1= get_model(self.model_key, rank=rank)
                inputs = torch.tensor(y).to(next(ASR_Emo_model.parameters()).device)
                with torch.no_grad():
                    output_ASR_emo = ASR_Emo_model.inference(
                        data_in=inputs,
                        language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
                        use_itn=False,
                        **kwargs1,
                    )
                
                video_audio_tags.append(output_ASR_emo[0][0]['text'].split('<|',2)[-1].split('|>')[0])
            else:
                video_audio_tags.append('')
            
        sample[Fields.speech_emotion] = video_audio_tags
        gc.collect()
        torch.cuda.empty_cache()
        return sample
