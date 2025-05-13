from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import argparse
import os
import json
import tqdm
from PIL import Image
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen2_5_model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--polished_prompt_data_json_path', type=str, default="./data_json.json")
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
    image_id_to_polished_caption = {}
    image_id_to_character_attributes = {}
    image_id_to_character_locations = {}
    image_id_to_scene_attributes = {}
    new_json = []

    pipe = pipeline(args.qwen2_5_model_path, backend_config=TurbomindEngineConfig(tp=args.gpu_nums))


    with open(args.polished_prompt_data_json_path, "r") as f:
        data = json.load(f)
        for temp_line in data:
            image_id_to_ori_caption[temp_line["image_id"]] = temp_line["ori_prompt"]
            image_id_to_polished_caption[temp_line["image_id"]] = temp_line["polished_prompt"]
            image_id_list.append(temp_line["image_id"])


    with open(args.character_attributes_data_json_path, "r") as f:
        data = json.load(f)
        for temp_line in data:
            image_id_to_character_attributes[temp_line["image_id"]] = temp_line["main_character_with_characteristics_list"]

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

        # filter character_attributes
        new_character_attributes = []
        for temp_character_attribute_list in image_id_to_character_attributes[temp_image_id]:

            temp_character_characteristics_list = []
            for temp_characteristic in temp_character_attribute_list["characteristics_list"]:
                temp_characteristic_dict = {}
                temp_characteristic_dict["main_character"] = temp_character_attribute_list["main_character"]
                temp_characteristic_dict["characteristic"] = temp_characteristic

                prompt = "I will provide you with a paragraph of image description and a specific characteristic of a main character. Your task is to determine whether the given image description includes the description of this particular characteristic of the main character. Below is the image description text: \'"
                prompt += image_id_to_polished_caption[temp_image_id]
                prompt += "\', and below is the specific characteristic of the main character: \'"
                prompt += str(temp_characteristic_dict)
                prompt += "\'. Please determine whether the provided image description includes description about this particular characteristic of the designated main character. Please respond with only 'yes' or 'no'."
                
                output_text = pipe((prompt), gen_config=GenerationConfig(max_new_tokens=16))
                response = output_text.text


                if 'yes' in response:
                    temp_character_characteristics_list.append(temp_characteristic)

            if len(temp_character_characteristics_list) == 0:
                continue
            else:
                temp_character_characteristics_json = {}
                temp_character_characteristics_json["main_character"] = temp_character_attribute_list["main_character"]
                temp_character_characteristics_json["characteristics_list"] = temp_character_characteristics_list
                temp_character_characteristics_json["cls"] = temp_character_attribute_list["cls"]
                new_character_attributes.append(temp_character_characteristics_json)
        

        # filter character_locations
        new_character_locations = []
        for temp_character_location in image_id_to_character_locations[temp_image_id]:
            if temp_character_location["position"] == None:
                continue

            temp_location_dict = {}
            temp_location_dict["main_character"] = temp_character_location["main_character"]
            temp_location_dict["position"] = temp_character_location["position"]

            prompt = "I will provide you with a paragraph of image description and a specific positional feature of a main character. Your task is to determine whether the given image description includes the description of this particular positional feature of the main character. Below is the image description text: \'"
            prompt += image_id_to_polished_caption[temp_image_id]
            prompt += "\', and below is the specific positional feature of the main character: \'"
            prompt += str(temp_location_dict)
            prompt += "\'. Please determine whether the provided image description includes description about this particular positional feature of the designated main character. Please respond with only 'yes' or 'no'."
            
            output_text = pipe((prompt), gen_config=GenerationConfig(max_new_tokens=16))
            response = output_text.text

            if 'yes' in response:
                new_character_locations.append(temp_character_location)
        
        # filter scene_attributes
        new_scene_attributes = []
        for temp_scene_attributes in image_id_to_scene_attributes[temp_image_id]:
            if temp_scene_attributes == "spatial":
                continue
            temp_scene_attributes_dict = {}
            temp_scene_attributes_dict["scene_attribute"] = temp_scene_attributes
            temp_scene_attributes_dict["content"] = image_id_to_scene_attributes[temp_image_id][temp_scene_attributes]

            prompt = "I will provide you with a paragraph of image description and a specific scene attribute of the image. Your task is to determine whether the given image description includes the description of this particular scene attribute. Below is the image description text: \'"
            prompt += image_id_to_polished_caption[temp_image_id]
            prompt += "\', and below is the scene attribute: \'"
            prompt += str(temp_scene_attributes_dict)
            prompt += "\'. Please determine whether the provided image description includes description about this particular scene attribute of the image. Please respond with only 'yes' or 'no'."
            
            output_text = pipe((prompt), gen_config=GenerationConfig(max_new_tokens=16))
            response = output_text.text

            if 'yes' in response:
                new_scene_attributes.append(temp_scene_attributes_dict)

        # filter scene_attributes spatial
        try:
            temp_saptial_list = []
            if isinstance(image_id_to_scene_attributes[temp_image_id]["spatial"], list):
                for temp_spatial in image_id_to_scene_attributes[temp_image_id]["spatial"]:
                    temp_scene_attributes_dict = {}
                    temp_scene_attributes_dict["scene_attribute"] = "spatial"
                    temp_scene_attributes_dict["content"] = temp_spatial

                    prompt = "I will provide you with a paragraph of image description and a specific scene attribute of the image. Your task is to determine whether the given image description includes the description of this particular scene attribute. Below is the image description text: \'"
                    prompt += image_id_to_polished_caption[temp_image_id]
                    prompt += "\', and below is the scene attribute: \'"
                    prompt += str(temp_scene_attributes_dict)
                    prompt += "\'. Please determine whether the provided image description includes description about this particular scene attribute of the image. Please respond with only 'yes' or 'no'."
                    
                    output_text = pipe((prompt), gen_config=GenerationConfig(max_new_tokens=16))
                    response = output_text.text
                    
                    if 'yes' in response:
                        temp_saptial_list.append(temp_spatial)
            else:
                temp_spatial = image_id_to_scene_attributes[temp_image_id]["spatial"]
                temp_scene_attributes_dict = {}
                temp_scene_attributes_dict["scene_attribute"] = "spatial"
                temp_scene_attributes_dict["content"] = temp_spatial

                prompt = "I will provide you with a paragraph of image description and a specific scene attribute of the image. Your task is to determine whether the given image description includes the description of this particular scene attribute. Below is the image description text: \'"
                prompt += image_id_to_polished_caption[temp_image_id]
                prompt += "\', and below is the scene attribute: \'"
                prompt += str(temp_scene_attributes_dict)
                prompt += "\'. Please determine whether the provided image description includes description about this particular scene attribute of the image. Please respond with only 'yes' or 'no'."
                
                output_text = pipe((prompt), gen_config=GenerationConfig(max_new_tokens=16))
                response = output_text.text
                
                if 'yes' in response:
                    temp_saptial_list.append(temp_spatial)


        except:
            temp_saptial_list = []

        if not len(temp_saptial_list) == 0:
            new_scene_attributes.append({"scene_attribute":"spatial", "content":temp_saptial_list})


        temp_new_polished_prompt_json = {}
        temp_new_polished_prompt_json["image_id"] = temp_image_id
        temp_new_polished_prompt_json["ori_prompt"] = image_id_to_ori_caption[temp_image_id]
        temp_new_polished_prompt_json["polished_prompt"] = image_id_to_polished_caption[temp_image_id]
        temp_new_polished_prompt_json["character_attributes"] = new_character_attributes
        temp_new_polished_prompt_json["character_locations"] = new_character_locations
        temp_new_polished_prompt_json["scene_attributes"] = new_scene_attributes
        
        print(temp_new_polished_prompt_json)

        new_json.append(temp_new_polished_prompt_json)

    with open(args.output_json, "a") as f:
        json.dump(new_json, f)