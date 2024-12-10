import copy
import random
import threading

import numpy as np
from jsonargparse.typing import PositiveInt
from loguru import logger
from PIL import ImageOps
from data_juicer.utils.constant import Fields
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import HashKeys
from data_juicer.utils.mm_utils import (SpecialTokens, extract_key_frames,
                                        extract_video_frames_uniformly,
                                        insert_texts_after_placeholders,
                                        load_data_with_context, load_video,
                                        remove_non_special_tokens,
                                        remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
import sys
sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/ShareGPT4Video')
from data_juicer.my_pretrained_method.ShareGPT4Video.run import single_test
import gc, time

class TimeoutException(Exception):
    pass

def single_test_with_timeout(model, processor, tokenizer, video_id_now, qs, pre_query_prompt, num_frames, conv_mode, timeout=10):
    result = []
    exception = []

    def target():
        try:
            output = single_test(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                vid_path = video_id_now,
                qs=qs,
                pre_query_prompt=pre_query_prompt,
                num_frames=num_frames,
                conv_mode=conv_mode
            )
            result.append(output)
        except Exception as e:
            exception.append(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutException(f"Operation timed out after {timeout} seconds")
    if exception:
        raise exception[0]

    return result[0]


OP_NAME = 'video_captioning_mapper_T'

with AvailabilityChecking(['torch', 'transformers'],
                          OP_NAME):

    import torch, os, tempfile, shutil
    import pickle, copy
    import transformers  # noqa: F401
    
    # avoid hanging when calling clip in multiprocessing
    torch.set_num_threads(1)

class ProcessingTimeoutException(Exception):
    def __init__(self, video_name, message="Processing time exceeded 60 seconds"):
        self.video_name = video_name
        self.message = f"{message} for video: {video_name}"
        super().__init__(self.message)


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningMapperT(Mapper):
    """Mapper to generate samples whose captions are generated based on
    a video-to-text model and sampled video frame."""

    def __init__(
        self,
        query: str = "Please describe the video, including its event, environment and atmosphere. Less than 200 words.",
        video_describe_model_path: str = '/mnt1/daoyuan_mm/sharegpt4video-8b',
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_video_blip: video-blip model name on huggingface
            to generate caption
        """
        super().__init__(*args, **kwargs)

        self._batched_op = True
        self._accelerator = 'cuda'
        self.query = query
        self.pre_query_prompt = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. Highlighting any significant events, characters, or objects that appear throughout the frames."
    
        self.model_key = prepare_model(
            model_type='sharegpt4video',
            pretrained_model_name_or_path=video_describe_model_path,
        )

    def process(self, samples, rank=None, context=False):

        Total_information = []
        loaded_video_keys = samples[self.video_key][0]

        tokenizer, model, processor = get_model(self.model_key, rank=rank)

        for vedio_id,video_keys in enumerate(loaded_video_keys):
            try:
                outputs = single_test_with_timeout(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    video_id_now=video_keys,
                    qs=self.query,
                    pre_query_prompt=self.pre_query_prompt,
                    num_frames=16,
                    conv_mode='llava_llama_3',
                    timeout=20  # 设置超时时间为10秒
                )
            except TimeoutException as e:
                print(video_keys + f"  Error: {e}")
                Total_information.append(["False"])
                break

            Total_information.append([outputs])

        samples[Fields.video_caption] = [Total_information]

        # Fields.track_video_caption
        gc.collect()
        torch.cuda.empty_cache()
        return samples
