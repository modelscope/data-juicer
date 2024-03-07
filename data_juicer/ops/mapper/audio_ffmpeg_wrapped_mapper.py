from typing import Dict, List, Optional

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.logger_utils import HiddenPrints

from ..base_op import OPERATORS, Mapper

OP_NAME = 'audio_ffmpeg_wrapped_mapper'

with AvailabilityChecking(['ffmpeg-python'], OP_NAME), HiddenPrints():
    import ffmpeg


@OPERATORS.register_module(OP_NAME)
class AudioFFmpegWrappedMapper(Mapper):
    """Simple wrapper for FFmpeg audio filters.
    """

    def __init__(
        self,
        filter_name: Optional[str] = None,
        filter_kwargs: Optional[Dict] = None,
        global_args: Optional[List[str]] = None,
        capture_stderr: bool = True,
        overwrite_output: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param filter_name: ffmpeg audio filter name.
        :param filter_kwargs: keyword-arguments passed to ffmpeg filter.
        :param global_args: list-arguments passed to ffmpeg command-line.
        :param capture_stderr: whether to capture stderr.
        :param overwrite_output: whether to overwrite output file.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.filter_name = filter_name
        self.filter_kwargs = filter_kwargs
        self.global_args = global_args
        self.capture_stderr = capture_stderr
        self.overwrite_output = overwrite_output

    def process(self, sample):
        # there is no audio in this sample
        if self.audio_key not in sample or not sample[self.audio_key]:
            return sample

        if self.filter_name is None:
            return sample

        loaded_audio_keys = sample[self.audio_key]
        proceessed = {}
        for audio_key in loaded_audio_keys:
            if audio_key in proceessed:
                continue

            output_key = transfer_filename(audio_key, OP_NAME,
                                           **self._init_parameters)
            stream = (ffmpeg.input(audio_key).filter(
                self.filter_name, **self.filter_kwargs).output(output_key))
            if self.global_args is not None:
                stream = stream.global_args(*self.global_args)
            stream.run(capture_stderr=self.capture_stderr,
                       overwrite_output=self.overwrite_output)
            proceessed[audio_key] = output_key

        sample[self.audio_key] = [proceessed[key] for key in loaded_audio_keys]
        return sample
