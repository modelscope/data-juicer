import copy
import random

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
from deepface import DeepFace
from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
from data_juicer.utils.ASD_mapper_utils import get_video_array_cv2, annotate_video_with_bounding_boxes, crop_from_array
import sys
sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/ShareGPT4Video')
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from data_juicer.my_pretrained_method.ShareGPT4Video.run import single_test
import gc

OP_NAME = 'video_captioning_face_attribute_emotion_mapper'

with AvailabilityChecking(['torch', 'transformers'],
                          OP_NAME):

    import torch, os, tempfile, shutil
    from shutil import rmtree
    import pickle, copy, cv2
    import transformers  # noqa: F401
    
    # avoid hanging when calling clip in multiprocessing
    torch.set_num_threads(1)
import sys
sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/VideoLLaMA2')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init



@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningFaceAttributeEmotionMapper(Mapper):
    """Mapper to generate samples whose captions are generated based on
    a video-to-text model and sampled video frame."""

    def __init__(
        self,
        face_track_query: str = "Please describe the person's facial expression, tell me the person's emotion through the video, like Happiness, Excitement, Love, Gratitude, Relief, Pride, Anger, Sadness, Fear, Guilt, Shame, Disgust, Surprise, Confusion, Curiosity, Boredom ...",
        cropping_face_video_tempt_path = './tempt_video/tmp_video_remove',
        video_describe_model_path: str = 'pt_model/VideoLLaMA2',
        *args,
        **kwargs
    ):
        """
        Initialization method.

        :param hf_video_blip: video-blip model name on huggingface
            to generate caption
        """
        super().__init__(*args, **kwargs)

        self._batched_op = True
        self._accelerator = 'cuda'
        self.context_param = 0.8

        # self.pre_query_prompt = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. "
        self.query = face_track_query
        self.cropping_face_video_tempt_path = cropping_face_video_tempt_path

        self.model_key = prepare_model(
            model_type='VideoLLaMA2',
            pretrained_model_name_or_path=video_describe_model_path,
        )



    def process(self, samples, rank=None):

        Total_information = []
        video_samples = samples[Fields.human_track_data_path]
        loaded_video_keys = samples[self.video_key][0]
        
        cropping_face_video_tempt_path = tempfile.mkdtemp(dir=self.cropping_face_video_tempt_path)
        if os.path.exists(cropping_face_video_tempt_path):
            rmtree(cropping_face_video_tempt_path)

        os.makedirs(cropping_face_video_tempt_path, exist_ok = False)
        model, processor, tokenizer= get_model(self.model_key, rank=rank)
        for vedio_id,ASD_attribute_all_tracks_for_one_video in enumerate(video_samples[0]):
            if len(ASD_attribute_all_tracks_for_one_video) == 0:
                Total_information.append([])
                continue
            
            description_for_each_track = []
            video_array = get_video_array_cv2(loaded_video_keys[vedio_id])
            for track_id,tracks_now in enumerate(ASD_attribute_all_tracks_for_one_video):
                cs = self.context_param  

                with open(tracks_now['bbox_path'], 'rb') as f:
                    bbox_data = pickle.load(f)
                    xys_bbox = bbox_data['xys_bbox']
                    track_frame = bbox_data['frame']

                face_video_out_path = os.path.join(cropping_face_video_tempt_path, loaded_video_keys[vedio_id].split('/')[-1][:-4] + '__' + str(track_id) + '.mp4')
                vOut = cv2.VideoWriter(face_video_out_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	
                start_frame_id_in = 0
                start_frame_id_out = track_frame[start_frame_id_in]  # tag
                while start_frame_id_in + 1 <len(track_frame):
                    bs = xys_bbox['s'][start_frame_id_in]  
                    bsi = int(bs * (1 + 2 * cs))  

                    start_frame_id_in = start_frame_id_in + 1
                    start_frame_id_out = track_frame[start_frame_id_in]
                    frame_before_crop = video_array[start_frame_id_out]

                    frame = np.pad(frame_before_crop, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
                    my  = xys_bbox['y'][start_frame_id_in] + bsi  # BBox center Y
                    mx  = xys_bbox['x'][start_frame_id_in] + bsi  # BBox center X
                    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
                    vOut.write(cv2.resize(face, (224, 224)))
                vOut.release()
                
                outputs = mm_infer(processor['video'](face_video_out_path), self.query, model=model, tokenizer=tokenizer, do_sample=False, modal='video')

                description_for_each_track.append(outputs)
            
            Total_information.append(description_for_each_track)

        shutil.rmtree(cropping_face_video_tempt_path)
        samples[Fields.video_facetrack_attribute_emotion] = [Total_information]
        gc.collect()
        torch.cuda.empty_cache()
        return samples