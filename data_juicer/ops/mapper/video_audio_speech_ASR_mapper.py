import librosa
from data_juicer.utils.mm_utils import extract_audio_from_video
from data_juicer.utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Mapper
import gc
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = 'video_audio_speech_ASR_mapper'

import torch
torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
class VideoAudioSpeechASRMapper(Mapper):
    """Mapper to generate video tags from audio streams extracted by video
    using the Audio Spectrogram Transformer.
    """
    _accelerator = 'cuda'
    _batched_op = True

    def __init__(self,
                 model_dir_ASR = 'FunAudioLLM/SenseVoiceSmall',
                 speech_ASR: str = MetaKeys.speech_ASR,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        kwargs.setdefault('mem_required', '20GB')
        super().__init__(*args, **kwargs)
        self._batched_op = True
        self._model_sampling_rate = 16000
        self.model_dir_ASR = model_dir_ASR

        self.model_key = prepare_model(
            model_type='SenseVoiceSmall',
            pretrained_model_name_or_path=model_dir_ASR,
        )

        self.speech_ASR = speech_ASR

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if MetaKeys.speech_emotion in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample
        
        if not MetaKeys.video_audio_tags in sample[Fields.meta]:
            raise ValueError("video_active_speaker_mapper must be operated after video_tagging_from_audio_mapper.")


        # load video paths
        loaded_video_keys = sample[self.video_key]
        audio_tags = sample[Fields.meta][MetaKeys.video_audio_tags]

        ASR_model, kwargs1= get_model(self.model_key, rank=rank)

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
            
        sample[Fields.meta][self.speech_ASR] = video_audio_tags
        gc.collect()
        torch.cuda.empty_cache()
        return sample
