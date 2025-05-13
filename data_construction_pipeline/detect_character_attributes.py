from lmdeploy import pipeline, TurbomindEngineConfig
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
    parser.add_argument('--qwen2_5_vl_model_path', type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument('--data_json_path', type=str, default="./data_json.json")
    parser.add_argument('--image_folder', type=str, default="./image_folder")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--gpu_nums', type=int, default=1)
    args=parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    pipe = pipeline(args.qwen2_5_vl_model_path, backend_config=TurbomindEngineConfig(tp=args.gpu_nums))

    new_json = []
    broken_num = 0

    with open(args.data_json_path, "r") as f:
        data = json.load(f)
    
    for temp_line in tqdm.tqdm(data):
        character_to_characteristics  = {}
        character_to_cls = {}

        for temp_character in temp_line["all_main_character"]:
            
            # detect class
            prompt = "Please classify the character \"" + temp_character + "\" into the following categories: ['object', 'animal', 'person', 'text', 'other']. Only reply with the most fitting single category."
            output_text = pipe((prompt))
            output_text = output_text.text

            character_to_cls[temp_character] = output_text

            # detect feature
            prompt = "I will provide you with the corresponding description of an image, as follows: \"" + temp_line["ori_caption"] + "\" Please extract all descriptions of the features related to \'" + temp_character + "\' from this text, which may include color, material, action, and other typical features, and compile them into a list of phrase string. Return only the phrase string list."
            output_text = pipe((prompt))
            output_text = output_text.text
            
            try:
                character_to_characteristics[temp_character] = eval(output_text)
            except:
                print("not_list")
                character_to_characteristics[temp_character] = output_text


        
        image = Image.open(os.path.join(args.image_folder, temp_line["image_id"]))

        for temp_character_with_bbox_idx, temp_character_with_bbox in enumerate(temp_line["main_character_list"]):
            
            crop_img = image.crop(temp_character_with_bbox["bbox"])
            new_temp_character_json = {}
            
            cache_img_path = "temp_" + str(random.randint(0,9999)) + "_" + str(temp_character_with_bbox_idx) + temp_line["image_id"]
            crop_img.save(cache_img_path)

            try:
                temp_character_cls = character_to_cls[temp_character_with_bbox["main_character"]]
            except:
                broken_num += 1
                os.remove(cache_img_path)
                continue

            
            if "object" in temp_character_cls:
                image = load_image(cache_img_path)
                prompt = "Please analyze the key characteristics of the main object in this image, specifically the \'" + temp_character_with_bbox["main_character"] + "\', which may include color, material, shape, and other typical features. Currently identified characteristics include \"" + str(temp_character_cls) + "\". Please expand this list and respond in an identically formatted phrase string list."
                output_text = pipe((prompt, image))
                output_text = output_text.text
                

            elif "animal" in temp_character_cls:
                image = load_image(cache_img_path)
                prompt = "Please analyze the key characteristics of the primary animal in this image, specifically the \'" + temp_character_with_bbox["main_character"] + "\', which may include color, action, and other typical features. Currently identified characteristics include \"" + str(temp_character_cls) + "\". Please expand this list and respond in an identically formatted phrase string list."
                output_text = pipe((prompt, image))
                output_text = output_text.text
                

            elif "person" in temp_character_cls:
                image = load_image(cache_img_path)
                prompt = "Please analyze the key characteristics of the primary person in this image, specifically the \'" + temp_character_with_bbox["main_character"] + "\', which may include clothing, ages, and other typical features. Currently identified characteristics include \"" + str(temp_character_cls) + "\". Please expand this list and respond in an identically formatted phrase string list."
                output_text = pipe((prompt, image))
                output_text = output_text.text
            

            elif "text" in temp_character_cls:
                image = load_image(cache_img_path)
                prompt = "Please analyze the key characteristics of the primary text in this image, specifically the \'" + temp_character_with_bbox["main_character"] + "\', which may include color, content, font, and other typical features. Currently identified characteristics include \"" + str(temp_character_cls) + "\". Please expand this list and respond in an identically formatted phrase string list."
                output_text = pipe((prompt, image))
                output_text = output_text.text
                

            else:
                print("other cls")
                image = load_image(cache_img_path)
                prompt = "Please analyze the key characteristics of the primary character in this image, specifically the \'" + temp_character_with_bbox["main_character"] + "\'. Currently identified characteristics include \"" + str(temp_character_cls) + "\". Please expand this list and respond in an identically formatted phrase string list."
                output_text = pipe((prompt, image))
                output_text = output_text.text

            

            final_characteristic_list = []
            # filter
            try:
                characteristic_list = eval(output_text)
            except:
                characteristic_list = output_text

            if isinstance(characteristic_list, list):
                if len(characteristic_list) == 1:
                    characteristic_list = characteristic_list[0].replace("_", " ").split(", ")

                try:
                    for temp_characteristic in characteristic_list:

                        image = load_image(cache_img_path)
                        prompt = "Please analyze the main character in this image, specifically the \"" + temp_character_with_bbox["main_character"] + "\". Is \"" + temp_characteristic + "\" one of its features? Only respond with 'yes' if it is a perfect match. Please only respond with 'yes' or 'no'."
                        output_text = pipe((prompt, image))
                        output_text = output_text.text

                        if 'yes' in output_text:
                            final_characteristic_list.append(temp_characteristic)
                except:
                    os.remove(cache_img_path)
                    continue
            else:
                try:
                    characteristic_list = output_text.split("\n")
                    if len(characteristic_list) == 1:
                        characteristic_list = characteristic_list[0].replace("_", " ").split(", ")

                    for temp_characteristic in characteristic_list:
                        image = load_image(cache_img_path)
                        prompt = "Please analyze the main character in this image, specifically the \"" + temp_character_with_bbox["main_character"] + "\". Is \"" + temp_characteristic + "\" one of its features? Only respond with 'yes' if it is a perfect match. Please only respond with 'yes' or 'no'."
                        output_text = pipe((prompt, image))
                        output_text = output_text.text
                        
                        if 'yes' in output_text:
                            final_characteristic_list.append(temp_characteristic)
                except:
                    os.remove(cache_img_path)
                    continue

            if len(final_characteristic_list) == 0:
                os.remove(cache_img_path)
                continue
            else:
                print(final_characteristic_list)
                character_to_characteristics[temp_character_with_bbox["main_character"]] = final_characteristic_list
        
            os.remove(cache_img_path)


        new_character_list = []
        for temp_character in temp_line["all_main_character"]:
            temp_character_json = {}
            temp_character_json["main_character"] = temp_character
            temp_character_json["characteristics_list"] = character_to_characteristics[temp_character]
            temp_character_json["cls"] = character_to_cls[temp_character]
            new_character_list.append(temp_character_json)
  
        new_sample_json = {}
        new_sample_json["image_id"] = temp_line["image_id"]
        new_sample_json["dataset_target"] = temp_line["dataset_target"]
        new_sample_json["ori_caption"] = temp_line["ori_caption"]
        new_sample_json["main_character_with_characteristics_list"] = new_character_list

        new_json.append(new_sample_json)

    with open(args.output_json, "a") as f:
        json.dump(new_json, f)

    print("broken num: " + str(broken_num))