import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_output_log_dir_name', type=str, default="./playground/evaluation")
    parser.add_argument('--name_prefix', type=str, default="your_model_s_name")
    args=parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    dir_name = args.eval_output_log_dir_name
    file_name = os.listdir(dir_name)

    object_presence_all_count = 0
    object_presence_success_count = 0
    character_attributes_all_count = {"animal":0, "object":0, "person":0}
    character_attributes_success_count = {"animal":0, "object":0, "person":0}
    character_locations_all_count = 0
    character_locations_success_count = 0
    scene_attrbutes_all_count = {"background":0, "light":0, "style":0, "spatial":0}
    scene_attrbutes_success_count = {"background":0, "light":0, "style":0, "spatial":0}
    
    for temp_name in file_name:
        if args.name_prefix in temp_name and "overall" in temp_name:
            print(temp_name)
            with open(os.path.join(dir_name, temp_name), "r") as f:
                temp_json = json.load(f)
                object_presence_all_count += temp_json["object_presence_all_count"]
                object_presence_success_count += temp_json["object_presence_success_count"]
                character_attributes_all_count["animal"] += temp_json["character_attributes_all_count"]["animal"]
                character_attributes_all_count["object"] += temp_json["character_attributes_all_count"]["object"]
                character_attributes_all_count["person"] += temp_json["character_attributes_all_count"]["person"]
                character_attributes_success_count["animal"] += temp_json["character_attributes_success_count"]["animal"]
                character_attributes_success_count["object"] += temp_json["character_attributes_success_count"]["object"]
                character_attributes_success_count["person"] += temp_json["character_attributes_success_count"]["person"]
                character_locations_all_count += temp_json["character_locations_all_count"]
                character_locations_success_count += temp_json["character_locations_success_count"]
                scene_attrbutes_all_count["background"] += temp_json["scene_attrbutes_all_count"]["background"]
                scene_attrbutes_all_count["light"] += temp_json["scene_attrbutes_all_count"]["light"]
                scene_attrbutes_all_count["style"] += temp_json["scene_attrbutes_all_count"]["style"]
                scene_attrbutes_all_count["spatial"] += temp_json["scene_attrbutes_all_count"]["spatial"]
                scene_attrbutes_success_count["background"] += temp_json["scene_attrbutes_success_count"]["background"]
                scene_attrbutes_success_count["light"] += temp_json["scene_attrbutes_success_count"]["light"]
                scene_attrbutes_success_count["style"] += temp_json["scene_attrbutes_success_count"]["style"]
                scene_attrbutes_success_count["spatial"] += temp_json["scene_attrbutes_success_count"]["spatial"]

    acc = {}
    acc["object_presence"] = round(object_presence_success_count/object_presence_all_count, 4)
    acc["character_attributes_animal"] = round(character_attributes_success_count["animal"]/character_attributes_all_count["animal"], 4)
    acc["character_attributes_object"] = round(character_attributes_success_count["object"]/character_attributes_all_count["object"], 4)
    acc["character_attributes_person"] = round(character_attributes_success_count["person"]/character_attributes_all_count["person"], 4)
    acc["character_locations"] = round(character_locations_success_count/character_locations_all_count, 4)
    acc["scene_attrbutes_background"] = round(scene_attrbutes_success_count["background"]/scene_attrbutes_all_count["background"], 4)
    acc["scene_attrbutes_light"] = round(scene_attrbutes_success_count["light"]/scene_attrbutes_all_count["light"], 4)
    acc["scene_attrbutes_style"] = round(scene_attrbutes_success_count["style"]/scene_attrbutes_all_count["style"], 4)
    acc["scene_attrbutes_spatial"] = round(scene_attrbutes_success_count["spatial"]/scene_attrbutes_all_count["spatial"], 4)


    print(acc)





