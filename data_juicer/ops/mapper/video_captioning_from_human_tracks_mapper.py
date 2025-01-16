import numpy as np
from data_juicer.utils.constant import Fields
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
from data_juicer.utils.ASD_mapper_utils import get_video_array_cv2
import sys
sys.path.append('./data_juicer/my_pretrained_method/ShareGPT4Video')
from data_juicer.my_pretrained_method.ShareGPT4Video.run import single_test
import gc

OP_NAME = 'video_captioning_from_human_tracks_mapper'

with AvailabilityChecking(['torch', 'transformers'],OP_NAME):

    import torch, os, tempfile, shutil
    from shutil import rmtree
    import pickle, cv2
    
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningFromHumanTracksMapper(Mapper):
    """Mapper to generate samples whose captions are generated based on
    a video-to-text model and sampled video frame."""

    def __init__(
        self,
        human_track_query: str = "Descibe the person's apperance. ",
        video_describe_model_path: str = 'pt_model/sharegpt4video-8b',
        tempt_video_path: str = None,
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

        self.pre_query_prompt = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. "
        self.query = human_track_query
        self.is_child_query = 'Is the person a little child? Just answer yes or no. '
        self.tempt_video_path = tempt_video_path

        self.model_key = prepare_model(
            model_type='sharegpt4video',
            pretrained_model_name_or_path=video_describe_model_path,
        )


    def process(self, samples, rank=None, context=False):

        Total_information = []
        Is_child = [] 
        video_samples = samples[Fields.human_track_data_path]
        loaded_video_keys = samples[self.video_key][0]

        cropping_face_video_tempt_path = tempfile.mkdtemp(dir=self.tempt_video_path)
        if os.path.exists(cropping_face_video_tempt_path):
            rmtree(cropping_face_video_tempt_path)
        
        os.makedirs(cropping_face_video_tempt_path, exist_ok = False)
        
        tokenizer, model, processor = get_model(self.model_key, rank=rank)

        for vedio_id,ASD_attribute_all_tracks_for_one_video in enumerate(video_samples[0]):
            if len(ASD_attribute_all_tracks_for_one_video) == 0:
                Total_information.append([])
                Is_child.append([])
                continue
            description_for_each_track = []
            is_child_for_each_track = []
            video_array = get_video_array_cv2(loaded_video_keys[vedio_id])
            for track_id,tracks_now in enumerate(ASD_attribute_all_tracks_for_one_video):
                with open(tracks_now['bbox_path'], 'rb') as f:
                    bbox_data = pickle.load(f)
                    xy_human_bbox = bbox_data['xy_human_bbox']
                    track_frame = bbox_data['frame']
                    
                face_video_out_path = os.path.join(cropping_face_video_tempt_path, loaded_video_keys[vedio_id].split('/')[-1][:-4] + '__' + str(track_id) + '.mp4')
                wide_max = int((np.array(xy_human_bbox['x2']) - np.array(xy_human_bbox['x1'])).max())
                height_max = int((np.array(xy_human_bbox['y2']) - np.array(xy_human_bbox['y1'])).max())
                pad_max_w = (wide_max - (np.array(xy_human_bbox['x2']) - np.array(xy_human_bbox['x1']))).max()
                pad_max_h = (height_max - (np.array(xy_human_bbox['y2']) - np.array(xy_human_bbox['y1']))).max()
                pad_max = int(max(pad_max_w,pad_max_h))+1

                start_frame_id_in = 0
                start_frame_id_out = track_frame[start_frame_id_in]  # tag

                vOut = cv2.VideoWriter(face_video_out_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (wide_max + 2,height_max+2))# Write video
	
                while start_frame_id_in + 1 <len(track_frame):
                    bsi = pad_max  #

                    start_frame_id_in = start_frame_id_in + 1
                    start_frame_id_out = track_frame[start_frame_id_in]
                    frame_before_crop = video_array[start_frame_id_out]

                    frame = np.pad(frame_before_crop, ((pad_max,pad_max), (pad_max,pad_max), (0, 0)), 'constant', constant_values=(110, 110))
                    
                    mx1_o  = int(xy_human_bbox['x1'][start_frame_id_in] + bsi)  # BBox center Y
                    mx2_o  = int(xy_human_bbox['x2'][start_frame_id_in] + bsi)  # BBox center X
                    my1_o  = int(xy_human_bbox['y1'][start_frame_id_in] + bsi)  # BBox center Y
                    my2_o  = int(xy_human_bbox['y2'][start_frame_id_in] + bsi)  # BBox center X

                    wide_target = mx2_o-mx1_o
                    high_target = my2_o-my1_o
                    left_add = int((wide_max + 2 - wide_target)/2)
                    right_add = wide_max + 2 - wide_target - left_add
                    # mx1 = mx1_o - left_add
                    # mx2 = mx2_o + right_add
                    up_add = int((height_max + 2 - high_target)/2)
                    down_add = height_max + 2 - high_target - up_add
                    # my1 = my1_o - up_add
                    # my2 = my2_o + down_add
                    human = frame[my1_o:my2_o,mx1_o:mx2_o]

                    padded_image = cv2.copyMakeBorder(
                        human,
                        up_add,
                        down_add,
                        left_add,
                        right_add,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )

                    vOut.write(padded_image)
                vOut.release()

                outputs = single_test(model,
                          processor,
                          tokenizer,
                          face_video_out_path,
                          qs=self.query,
                          pre_query_prompt=self.pre_query_prompt,
                          num_frames=16,
                          conv_mode='llava_llama_3')

                description_for_each_track.append([outputs])

                outputs_ischild_response = single_test(model,
                          processor,
                          tokenizer,
                          face_video_out_path,
                          qs=self.is_child_query,
                          pre_query_prompt=self.pre_query_prompt,
                          num_frames=16,
                          conv_mode='llava_llama_3')
                
                is_child_for_each_track.append([outputs_ischild_response])
            
            Total_information.append(description_for_each_track)
            Is_child.append(is_child_for_each_track)

        shutil.rmtree(cropping_face_video_tempt_path)
        samples[Fields.track_video_caption] = [Total_information]
        samples[Fields.video_facetrack_is_child] = [Is_child]
        # Fields.track_video_caption
        gc.collect()
        torch.cuda.empty_cache()
        return samples
