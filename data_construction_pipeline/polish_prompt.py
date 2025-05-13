from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
from qwen_vl_utils import process_vision_info
import argparse
import os
import json
import tqdm
from PIL import Image
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen2_5_model_path', type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument('--character_attributes_data_json_path', type=str, default="./data_json.json")
    parser.add_argument('--character_locations_data_json_path', type=str, default="./data_json.json")
    parser.add_argument('--scene_attributes_data_json_path', type=str, default="./data_json.json")
    parser.add_argument('--image_folder', type=str, default="./image_folder")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--gpu_nums', type=int, default=1)
    args=parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    image_id_list = []
    image_id_to_ori_caption = {}
    image_id_to_character_attributes = {}
    image_id_to_character_locations = {}
    image_id_to_scene_attributes = {}
    new_json = []

    pipe = pipeline(args.qwen2_5_model_path, backend_config=TurbomindEngineConfig(tp=args.gpu_nums))

    with open(args.character_attributes_data_json_path, "r") as f:
        data = json.load(f)
        for temp_line in data:
            image_id_to_ori_caption[temp_line["image_id"]] = temp_line["ori_caption"]
            image_id_to_character_attributes[temp_line["image_id"]] = temp_line["main_character_with_characteristics_list"]
            image_id_list.append(temp_line["image_id"])

    with open(args.character_locations_data_json_path, "r") as f:
        data = json.load(f)
        for temp_line in data:
            image_id_to_character_locations[temp_line["image_id"]] = temp_line["main_character_list"]

    with open(args.scene_attributes_data_json_path, "r") as f:
        data = json.load(f)
        for temp_line in data:
            temp_scene_attributes = {}
            temp_scene_attributes["background"] = temp_line["background"]
            temp_scene_attributes["light"] = temp_line["light"]
            temp_scene_attributes["style"] = temp_line["style"]
            temp_scene_attributes["spatial"] = temp_line["spatial"]
            image_id_to_scene_attributes[temp_line["image_id"]] = temp_scene_attributes

    print("start")

    for temp_image_id in tqdm.tqdm(image_id_list):

        # polish for character_attributes
        prompt = "I will provide you with a paragraph of image description and the characteristic information of the main characters in the paragraph. Some of the main characters' characteristics in the image description are missing. Your task is to supplement the missing main character information into the image description. Below is the image description text: \'"
        prompt += image_id_to_ori_caption[temp_image_id]
        prompt += "\', and below are the main characters along with their respective characteristics: \'"
        prompt += str(image_id_to_character_attributes[temp_image_id])
        prompt += "\'. Please help me fill in the missing main character information in the image description. Please respond only the modified image description."
        
        output_text = pipe((prompt), gen_config=GenerationConfig(max_new_tokens=2048))
        response = output_text.text



        try:
            # polish for character_locations
            valid_position = []
            for temp_position in image_id_to_character_locations[temp_image_id]:
                if temp_position["position"] == None:
                    continue

                temp_position_json = {}
                temp_position_json["main_character"] = temp_position["main_character"]
                temp_position_json["position"] = temp_position["position"]
                valid_position.append(temp_position_json)

            if not len(valid_position) == 0:
                prompt = "I will provide you with a paragraph of image description and the position information of the main characters in the paragraph. Some of the main characters' positions in the image description are missing. Your task is to supplement the missing position information into the image description. Below is the image description text: \'"
                prompt += response
                prompt += "\', and below are the main characters along with their respective positions: \'"
                prompt += str(valid_position)
                prompt += "\'. Please help me fill in the missing main character information in the image description. Please respond only the modified image description."
                
                output_text = pipe((prompt), gen_config=GenerationConfig(max_new_tokens=2048))
                response = output_text.text
                
        except:
            continue

        try:
            # polish for scene_attributes
            prompt = "I will provide you with a paragraph of image description and the scene attribute information of the described image. Some of the scene attribute information in the image description is missing. Your task is to supplement the missing scene attribute information into the image description. Below is the image description text: \'"
            prompt += response
            prompt += "\', and below are the scene attribute information: \'"
            prompt += str(image_id_to_scene_attributes[temp_image_id])
            prompt += "\'. Please help me fill in the missing scene attribute information in the image description. Please respond only the modified image description."
            
            output_text = pipe((prompt), gen_config=GenerationConfig(max_new_tokens=2048))
            response = output_text.text
            
        except:
            continue
            
        print(response)

        temp_polished_prompt_json = {}
        temp_polished_prompt_json["image_id"] = temp_image_id
        temp_polished_prompt_json["ori_prompt"] = image_id_to_ori_caption[temp_image_id]
        temp_polished_prompt_json["polished_prompt"] = response

        new_json.append(temp_polished_prompt_json)


    with open(args.output_json, "a") as f:
        json.dump(new_json, f)