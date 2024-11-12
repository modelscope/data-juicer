import librosa
import numpy as np

from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import AUTOINSTALL, LazyLoader
from data_juicer.utils.mm_utils import extract_audio_from_video
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

torch = LazyLoader('torch', 'torch')

OP_NAME = 'video_tagging_from_audio_mapper'


@OPERATORS.register_module(OP_NAME)
class VideoTaggingFromAudioMapper(Mapper):
    """Mapper to generate video tags from audio streams extracted by video
    using the Audio Spectrogram Transformer.
    """

    _accelerator = 'cuda'

    def __init__(self,
                 hf_ast: str = 'MIT/ast-finetuned-audioset-10-10-0.4593',
                 trust_remote_code: bool = False,
                 tag_field_name: str = Fields.video_audio_tags,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_ast: path to the HF model to tag from audios.
        :param trust_remote_code: whether to trust the remote code of HF models
        :param tag_field_name: the field name to store the tags. It's
            "__dj__video_audio_tags__" in default.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        AUTOINSTALL.check(['torchaudio'])
        self.model_key = prepare_model(model_type='huggingface',
                                       pretrained_model_name_or_path=hf_ast,
                                       trust_remote_code=trust_remote_code)
        self._model_sampling_rate = 16000
        self._no_audio_label = 'EMPTY'

        self.tag_field_name = tag_field_name

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if self.tag_field_name in sample:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[self.tag_field_name] = np.array([], dtype=np.str_)
            return sample

        # load video paths
        loaded_video_keys = sample[self.video_key]

        model, feature_extractor = get_model(self.model_key, rank,
                                             self.use_cuda())
        video_audio_tags = []
        for video_path in loaded_video_keys:
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
            inputs = feature_extractor(y,
                                       sampling_rate=sr,
                                       return_tensors='pt').to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_tag_id = torch.argmax(logits, dim=-1).item()
            predicted_tag = model.config.id2label[predicted_tag_id]
            video_audio_tags.append(predicted_tag)
        sample[self.tag_field_name] = np.array(video_audio_tags, dtype=np.str_)
        return sample
