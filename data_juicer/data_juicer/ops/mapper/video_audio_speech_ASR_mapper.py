import librosa

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import extract_audio_from_video
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
import gc


OP_NAME = 'video_audio_speech_ASR_mapper'

with AvailabilityChecking(['torch', 'transformers', 'torchaudio'], OP_NAME):
    import torch
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
class VideoAudioSpeechASRMapper(Mapper):
    """Mapper to generate video tags from audio streams extracted by video
    using the Audio Spectrogram Transformer.
    """

    def __init__(self,
                 model_dir_ASR='/mnt1/daoyuan_mm/SenseVoiceSmall',
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
        self.model_dir_ASR = model_dir_ASR

        self.model_key = prepare_model(
            model_type='SenseVoiceSmall',
            pretrained_model_name_or_path=self.model_dir_ASR,
        )

    def process(self, sample, rank=None):
        # check if it's generated already
        if Fields.speech_ASR in sample:
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
                
                ASR_model, kwargs1= get_model(self.model_key, rank=rank)
                inputs = torch.tensor(y).to(next(ASR_model.parameters()).device)
                with torch.no_grad():
                    output_ASR_emo = ASR_model.inference(
                        data_in=inputs,
                        language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
                        use_itn=False,
                        **kwargs1,
                    )
                
                video_audio_tags.append({'language':output_ASR_emo[0][0]['text'].split('<|',1)[-1].split('|>')[0], 'asr': output_ASR_emo[0][0]['text'].split('|>',4)[-1]})
            else:
                video_audio_tags.append('')
            
        sample[Fields.speech_ASR] = video_audio_tags
        gc.collect()
        torch.cuda.empty_cache()
        return sample
