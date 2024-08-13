import librosa

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import extract_audio_from_video
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

OP_NAME = 'video_tagging_from_audio_mapper'

with AvailabilityChecking(['torch', 'transformers', 'torchaudio'], OP_NAME):
    import torch
    import torchaudio  # noqa: F401
    import transformers  # noqa: F401

    # avoid hanging when calling recognizeAnything in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
class VideoTaggingFromAudioMapper(Mapper):
    """Mapper to generate video tags from audio streams extracted by video
    using the Audio Spectrogram Transformer.
    """

    _accelerator = 'cuda'

    def __init__(self,
                 hf_ast='MIT/ast-finetuned-audioset-10-10-0.4593',
                 trust_remote_code=False,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.model_key = prepare_model(model_type='huggingface',
                                       pretrained_model_name_or_path=hf_ast,
                                       trust_remote_code=trust_remote_code)
        self._model_sampling_rate = 16000
        self._no_audio_label = 'EMPTY'

    def process(self, sample, rank=None):
        # check if it's generated already
        if Fields.video_audio_tags in sample:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.video_audio_tags] = []
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
                                       return_tensors='pt')
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_tag_id = torch.argmax(logits, dim=-1).item()
            predicted_tag = model.config.id2label[predicted_tag_id]
            video_audio_tags.append(predicted_tag)
        sample[Fields.video_audio_tags] = video_audio_tags
        return sample
