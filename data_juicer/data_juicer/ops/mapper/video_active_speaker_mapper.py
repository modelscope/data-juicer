from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields

from data_juicer.utils.ASD_mapper_utils import get_video_array_cv2,evaluate_network, \
    crop_video_with_facetrack, longest_continuous_actives

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
from data_juicer.utils.model_utils import get_model, prepare_model
import gc,os

OP_NAME = 'video_active_speaker_mapper'

with AvailabilityChecking([], OP_NAME):
    import torch
    import sys
    sys.path.append('./data_juicer/my_pretrained_method/Light-ASD')
    import tempfile
    import shutil, pickle
    from shutil import rmtree
    import os, subprocess
    import tqdm, glob
    # from model.faceDetector.s3fd import S3FD


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoActiveSpeakerMapper(Mapper):
    """
    """

    _default_kwargs = {'upsample_num_times': 0}

    def __init__(self,
                 tempt_save_path: str = './HumanVBenchRecipe/dj_ASD_tempt',
                 face_track_bbox_path: str = './HumanVBenchRecipe/dj_human_track',
                 Light_ASD_model_path: str = 'weight/finetuning_TalkSet.model',
                 acitve_threshold: int = 15,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param blur_type: 
        """
        super().__init__(*args, **kwargs)
        self._accelerator = 'cuda'
        self._init_parameters = self.remove_extra_parameters(locals())
        self.acitve_threshold = acitve_threshold

        self.tempt_save_path = tempt_save_path
        self.face_track_bbox_path = face_track_bbox_path

        # Initialize ASD model
        self.ASD_model_key = prepare_model(model_type='Light_ASD',
                                       pretrained_model_name_or_path=Light_ASD_model_path)
    
    def active_speaker_detection_revise(self, active_score,is_child_descrip,speech_audio,face_gender):
        speech_child = speech_audio['child'][0]
        speech_male = speech_audio['male'][0]
        speech_female = speech_audio['female'][0]
        if speech_male > speech_female:
            speech_gender = 'Man'
            speech_gender_confidence = speech_male
        else:
            speech_gender = 'Woman'
            speech_gender_confidence = speech_female
        
        if ' not ' in is_child_descrip:
            is_child_apperance = False
        else:
            is_child_apperance = True

        if speech_child < 0.1:
            is_child_voice = False
        elif speech_audio['Age'][0]<=12:
            is_child_voice = True
        else:
            is_child_voice = 'Not Sure'

        # Consistency detection: only perform false positive detection on positive samples
        if active_score>self.acitve_threshold:
            speak_active = True
            # age consistency test:
            if not is_child_voice == 'Not Sure':
                if is_child_apperance == is_child_voice:
                    # gender consistency test
                    if speech_gender_confidence > 0.65 and float(face_gender[1]) > 0.65:
                        if not speech_gender == face_gender[0]:
                            speak_active = False
                else:
                    speak_active = False
            return speak_active
        else:
            return False
    
    
    def process(self, sample, rank=None):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample
        
        if not Fields.video_audio_tags in sample:
            raise ValueError("video_active_speaker_mapper must be operated after video_tagging_from_audio_mapper.")

        if not Fields.human_track_data_path in sample:
            raise ValueError("video_active_speaker_mapper must be operated after video_human_tracks_extraction_mapper.")

        if not Fields.audio_speech_attribute in sample:
            raise ValueError("video_active_speaker_mapper must be operated after audio_speech_attribute.")

        if not Fields.video_facetrack_attribute_demographic in sample:
            raise ValueError("video_active_speaker_mapper must be operated after video_facetrack_attribute_demographic.")

        if not Fields.video_facetrack_is_child in sample:
            raise ValueError("video_active_speaker_mapper must be operated after video_captioning_from_human_tracks_mapper.")

        loaded_video_keys = sample[self.video_key]
        audio_speech_attribute = sample[Fields.audio_speech_attribute]
        face_demographic = sample[Fields.video_facetrack_attribute_demographic][0]
        child_flag = sample[Fields.video_facetrack_is_child]

        Total_result = []

        temp_dir = tempfile.mkdtemp(dir=self.tempt_save_path)
        pyaviPath = os.path.join(temp_dir, 'pyavi')
        pyframesPath = os.path.join(temp_dir, 'pyframes')
        pyworkPath = os.path.join(temp_dir, 'pywork')
        pycropPath = os.path.join(temp_dir, 'pycrop')
        if os.path.exists(temp_dir):
            rmtree(temp_dir)

        audio_tag = sample[Fields.video_audio_tags]
        asd_detection_model = get_model(self.ASD_model_key, rank=rank)
        
        for id_out,video_key in enumerate(loaded_video_keys):
            os.makedirs(pyaviPath, exist_ok = False) # The path for the input video, input audio, output video
            os.makedirs(pyframesPath, exist_ok = False) # Save all the video frames
            os.makedirs(pyworkPath, exist_ok = False) # Save the results in this process by the pckl method
            os.makedirs(pycropPath, exist_ok = False) # Save the detected face clips (audio+video) in this process

            # Extract audio
            audio_is_empty = False
            audioFilePath = os.path.join(pyaviPath, 'audio.wav')
            command = ("ffmpeg -y -i '%s' -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
                (video_key, 10, audioFilePath))
            if audio_tag[id_out] == "EMPTY":
                audio_is_empty = True
            else:
                subprocess.call(command, shell=True, stdout=None)
            

            video_array = get_video_array_cv2(video_key)

            def load_pkl(file_path):
                with open(file_path, 'rb') as file:
                    return pickle.load(file)
            # get allTracks
            allTracks = [load_pkl(item['bbox_path']) for item in sample[Fields.human_track_data_path][id_out]]
            
            # Face clips cropping
            for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
                result = crop_video_with_facetrack(video_array, track, os.path.join(pycropPath, '%05d' % ii), audioFilePath, audio_is_empty)
                if not result:
                    raise ValueError("something wrong with crop_video_with_facetrack.")

            # Active Speaker Detection
            if audio_tag[id_out] == 'Speech':
                files = glob.glob("%s/*.avi"%pycropPath)
                files.sort()
                try:
                    scores = evaluate_network(files, asd_detection_model, pycropPath)
                except:
                    scores = [[-10000]]* len(allTracks)

            else:
                scores = [[-10000]]* len(allTracks)

            for id in range(len(scores)):
                allTracks[id]['active_scores'] = scores[id]

            update_track = allTracks
            # for validation
            # visualization(vidTracks, scores, video_array, pyaviPath)

            shutil.rmtree(temp_dir)

            speak_flag_for_tracks_in_a_video = []
            for track_idx,track_i  in enumerate(update_track):
                active_count = longest_continuous_actives(track_i['active_scores'])
                audio_attri = audio_speech_attribute[id_out][0]
                is_child_descrip = child_flag[id_out][track_idx][0]
                face_gender = face_demographic[id_out][track_idx]['gender']
                flag = self.active_speaker_detection_revise(active_count, is_child_descrip, audio_attri, face_gender)
                speak_flag_for_tracks_in_a_video.append(flag)


            Total_result.append(speak_flag_for_tracks_in_a_video)
            torch.cuda.empty_cache()
 
        sample[Fields.ASD_revise_flag] = Total_result

        gc.collect()
        torch.cuda.empty_cache()

        return sample
