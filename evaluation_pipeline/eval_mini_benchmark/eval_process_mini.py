import tqdm
import os
import json
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from ultralytics import YOLOE
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch
import argparse
from PIL import Image
from PIL import ImageDraw
import random
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen2_5_vl_model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--yoloe_model_path', type=str, default="yoloe-11l-seg.pt")
    parser.add_argument('--blip_model_path', type=str, default="Salesforce/blip-itm-large-flickr")
    parser.add_argument('--ann_json_path', type=str, default="./data_json.json")
    parser.add_argument('--image_folder', type=str, default="./image_folder")
    parser.add_argument('--image_info_json', type=str, default="./image_info.json")
    parser.add_argument('--output_log_dir', type=str, default="./output/")
    parser.add_argument('--output_name_prefix', type=str, default="output")
    args=parser.parse_args()

    return args


def iou_cal(bbox1, bbox2):

    min_x1 = min(bbox1[0], bbox2[0])
    max_x1 = max(bbox1[0], bbox2[0])
    min_y1 = min(bbox1[1], bbox2[1])
    max_y1 = max(bbox1[1], bbox2[1])

    max_x2 = max(bbox1[2], bbox2[2])
    min_x2 = min(bbox1[2], bbox2[2])
    max_y2 = max(bbox1[3], bbox2[3])
    min_y2 = min(bbox1[3], bbox2[3])

    if min_x2 - max_x1 < 0 or min_y2 - max_y1 < 0:
        return 0, 0, 0

    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    intersection_area = (min_x2-max_x1) * (min_y2-max_y1)
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area

    return iou, area1, area2


if __name__ == "__main__":
    args = parse_args()

    yoloe_model = YOLOE(args.yoloe_model_path)
    blip_processor = BlipProcessor.from_pretrained(args.blip_model_path)
    blip_model = BlipForImageTextRetrieval.from_pretrained(
        args.blip_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    pipe = pipeline(args.qwen2_5_vl_model_path)

    image_info_dict = {}
    with open(args.image_info_json, "r") as f:
        image_info_data = json.load(f)

        for temp_data_piece in image_info_data:
            if temp_data_piece["image_id"] in image_info_dict:
                # for more than one generated image per prompt (optional)
                image_info_dict[temp_data_piece["image_id"]].append(temp_data_piece["output_image_name"])
            else:
                image_info_dict[temp_data_piece["image_id"]] = [temp_data_piece["output_image_name"]]

    with open(args.ann_json_path, "r") as f:
        ann_data = json.load(f)

    new_json = []
    no_found_image_id = []
    new_keep_image_state = []

    character_attributes_all_count = {"object":0, "animal":0, "person":0}
    character_attributes_success_count = {"object":0, "animal":0, "person":0}
    character_attributes_detect_count = {"object":0, "animal":0, "person":0}

    character_locations_all_count = 0
    character_locations_success_count = 0

    scene_attrbutes_all_count = {"background":0, "light":0, "style":0, "spatial":0}
    scene_attrbutes_success_count = {"background":0, "light":0, "style":0, "spatial":0}

    for temp_piece in tqdm.tqdm(ann_data):

        temp_piece_image_id = temp_piece["dataset_target"] + "_" + temp_piece["image_id"]

        if not temp_piece_image_id in image_info_dict:
            no_found_image_id.append(temp_piece_image_id)
            continue

        now_prompt_image_info = image_info_dict[temp_piece_image_id]

        valid_character_list = []
        for temp_character in temp_piece["character_attributes"]:
            valid_character_list.append(temp_character["main_character"])

        for temp_character in temp_piece["character_locations"]:
            if not temp_character["main_character"] in valid_character_list:
                valid_character_list.append(temp_character["main_character"])
        

        for temp_image_name_id, temp_image_name in enumerate(now_prompt_image_info):
            
            try:
                now_image = Image.open(os.path.join(args.image_folder, temp_image_name))
            except:
                no_found_image_id.append(temp_piece_image_id)
                continue
            image_size, _ = now_image.size
            temp_image_state = {}
            temp_image_state["temp_image_name"] = temp_image_name
            temp_image_state["temp_image_id"] = temp_image_name_id
            temp_image_state["prompt_info"] = temp_piece

            if temp_piece["mini_bench_target"] in ["object", "animal", "person"]:

                # object presence
                valid_character_list_bbox = {}

                for temp_character in valid_character_list:
                    names = [temp_character]
                    yoloe_model.set_classes(names, yoloe_model.get_text_pe(names))
                    try:
                        results = yoloe_model.predict(os.path.join(args.image_folder, temp_image_name), verbose=False)
                    except:
                        Image.open(os.path.join(args.image_folder, temp_image_name)).convert("RGB").save(os.path.join(args.image_folder, temp_image_name))
                        results = yoloe_model.predict(os.path.join(args.image_folder, temp_image_name), verbose=False)
                    yoloe_bboxes = results[0].boxes.xyxy.tolist()

                    if len(yoloe_bboxes) == 0:
                        valid_character_list_bbox[temp_character] = None
                        continue
                    yoloe_bbox = [0,0,0,0]
                    yoloe_bbox[0] = int(yoloe_bboxes[0][0])
                    yoloe_bbox[1] = int(yoloe_bboxes[0][1])
                    yoloe_bbox[2] = math.ceil(yoloe_bboxes[0][2])
                    yoloe_bbox[3] = math.ceil(yoloe_bboxes[0][3])

                    image = load_image(os.path.join(args.image_folder, temp_image_name))
                    prompt = "Please only provide the bounding box coordinate (as a list) of the region \"" + temp_character + "\" describes. Do not include any JSON formatting or additional text in the response."
                    output_text = pipe((prompt, image))
                    output_text = output_text.text

                    try:
                        output_text = output_text.replace("json", "").replace("```", "")
                        llm_bbox = eval(output_text)
                        temp_iou, area1, area2 = iou_cal(yoloe_bbox, llm_bbox)
                    except:
                        valid_character_list_bbox[temp_character] = None
                        continue

                    
                    if temp_iou > 0.7:
                        if area1 > area2:
                            valid_character_list_bbox[temp_character] = yoloe_bbox
                        else:
                            valid_character_list_bbox[temp_character] = llm_bbox
                    else:
                        try:
                            yoloe_bbox_crop_img = now_image.crop(yoloe_bbox)
                            llm_bbox_crop_img = now_image.crop(llm_bbox)
                            image_pair = [yoloe_bbox_crop_img, llm_bbox_crop_img]
                            blip_process_input = blip_processor(image_pair, [temp_character["main_character"]], return_tensors="pt", padding=True).to(blip_model.device)
                        except:
                            valid_character_list_bbox[temp_character] = None
                            continue
                        cosine_score = blip_model(pixel_values=blip_process_input['pixel_values'], input_ids=blip_process_input['input_ids'], attention_mask=blip_process_input['attention_mask'], use_itm_head=False).itm_score
                        
                        # if cosine_score[0] < 0.4 and cosine_score[1] < 0.4:
                        #     valid_character_list_bbox[temp_character] = None
                        #     continue
                        
                        if cosine_score[0] > cosine_score[1]:
                            valid_character_list_bbox[temp_character] = yoloe_bbox
                        else:
                            valid_character_list_bbox[temp_character] = llm_bbox


                # character attributes
                temp_image_state["character_attributes_detail_score"] = []
                valid_character_list_character_attributes = {}
                detected_character_s_attribute_num = {}
                temp_character_attributes = temp_piece["character_attributes"]
                temp_character_attributes_dict = {}
                temp_character_cls_dict = {}
                for temp_character_attributes_piece in temp_character_attributes:
                    if not temp_character_attributes_piece["cls"] == temp_piece["mini_bench_target"]:
                        continue
                    temp_character_attributes_dict[temp_character_attributes_piece["main_character"]] = temp_character_attributes_piece["characteristics_list"]
                    temp_character_cls_dict[temp_character_attributes_piece["main_character"]] = temp_character_attributes_piece["cls"]

                for temp_character in valid_character_list_bbox:
                    if not temp_character in temp_character_attributes_dict:
                        continue
                    now_character_attribute = temp_character_attributes_dict[temp_character]
                    valid_character_list_character_attributes[temp_character] = 0
                    
                    temp_character_attributes_detail_score_json = {}
                    temp_character_attributes_detail_score_json["main_character"] = temp_character
                    temp_character_attributes_detail_score_json["attributes_list"] = []
                        
                    if valid_character_list_bbox[temp_character] == None:
                        for temp_attribute_piece in now_character_attribute:
                            temp_character_attributes_detail_score_json["attributes_list"].append({"attribute": temp_attribute_piece, "score": 0})
                        temp_image_state["character_attributes_detail_score"].append(temp_character_attributes_detail_score_json)
                        continue

                    try:
                        crop_img = now_image.crop(valid_character_list_bbox[temp_character])
                    except:
                        for temp_attribute_piece in now_character_attribute:
                            temp_character_attributes_detail_score_json["attributes_list"].append({"attribute": temp_attribute_piece, "score": 0})
                        temp_image_state["character_attributes_detail_score"].append(temp_character_attributes_detail_score_json)
                        continue

                    if temp_character_cls_dict[temp_character] not in detected_character_s_attribute_num:
                        detected_character_s_attribute_num[temp_character_cls_dict[temp_character]] = len(now_character_attribute)
                    else:
                        detected_character_s_attribute_num[temp_character_cls_dict[temp_character]] += len(now_character_attribute)


                    temp_attribute_count = 0
                    for temp_attribute_piece in now_character_attribute:
                        prompt = "Please analyze the main character in this image, specifically the \"" + temp_character + "\". Please determine whether \"" + temp_attribute_piece + "\" is one of its characteristics or is associated with it. Please only respond with 'yes' or 'no'."
                        output_text = pipe((prompt, crop_img))
                        output_text = output_text.text
                        
                        if 'yes' in output_text.lower():
                            valid_character_list_character_attributes[temp_character] += 1
                            temp_character_attributes_detail_score_json["attributes_list"].append({"attribute": temp_attribute_piece, "score": 1})
                        else:
                            temp_character_attributes_detail_score_json["attributes_list"].append({"attribute": temp_attribute_piece, "score": 0})
                    temp_image_state["character_attributes_detail_score"].append(temp_character_attributes_detail_score_json)



                # print(valid_character_list_character_attributes)
                # character attribute stastic
                temp_image_state["character_attribute_all"] = {}
                temp_image_state["character_attribute_success"] = {}
                temp_image_state["detected_character_s_attribute_num"] = detected_character_s_attribute_num

                for temp_detect_chharacter_cls in detected_character_s_attribute_num:
                    if not temp_detect_chharacter_cls in character_attributes_detect_count:
                        character_attributes_detect_count[temp_detect_chharacter_cls] = 0
                    character_attributes_detect_count[temp_detect_chharacter_cls] += detected_character_s_attribute_num[temp_detect_chharacter_cls]

                for temp_character in temp_character_attributes_dict:
                    if not temp_character_cls_dict[temp_character] in character_attributes_all_count:
                        character_attributes_all_count[temp_character_cls_dict[temp_character]] = 0
                    character_attributes_all_count[temp_character_cls_dict[temp_character]] += len(temp_character_attributes_dict[temp_character])

                    if not temp_character_cls_dict[temp_character] in temp_image_state["character_attribute_all"]:
                        temp_image_state["character_attribute_all"][temp_character_cls_dict[temp_character]] = 0
                    temp_image_state["character_attribute_all"][temp_character_cls_dict[temp_character]] += len(temp_character_attributes_dict[temp_character])


                for temp_character in valid_character_list_character_attributes:
                    if not temp_character_cls_dict[temp_character] in character_attributes_success_count:
                        character_attributes_success_count[temp_character_cls_dict[temp_character]] = 0   
                    character_attributes_success_count[temp_character_cls_dict[temp_character]] += valid_character_list_character_attributes[temp_character]

                    if not temp_character_cls_dict[temp_character] in temp_image_state["character_attribute_success"]:
                        temp_image_state["character_attribute_success"][temp_character_cls_dict[temp_character]] = 0
                    temp_image_state["character_attribute_success"][temp_character_cls_dict[temp_character]] += valid_character_list_character_attributes[temp_character]
                    
                new_keep_image_state.append(temp_image_state)

                print(character_attributes_all_count)
                print(character_attributes_success_count)
                print(character_attributes_detect_count)

            elif temp_piece["mini_bench_target"] == "location":

                # object presence
                valid_character_list_bbox = {}

                for temp_character in valid_character_list:
                    names = [temp_character]
                    yoloe_model.set_classes(names, yoloe_model.get_text_pe(names))
                    try:
                        results = yoloe_model.predict(os.path.join(args.image_folder, temp_image_name), verbose=False)
                    except:
                        Image.open(os.path.join(args.image_folder, temp_image_name)).convert("RGB").save(os.path.join(args.image_folder, temp_image_name))
                        results = yoloe_model.predict(os.path.join(args.image_folder, temp_image_name), verbose=False)
                    yoloe_bboxes = results[0].boxes.xyxy.tolist()

                    if len(yoloe_bboxes) == 0:
                        valid_character_list_bbox[temp_character] = None
                        continue
                    yoloe_bbox = [0,0,0,0]
                    yoloe_bbox[0] = int(yoloe_bboxes[0][0])
                    yoloe_bbox[1] = int(yoloe_bboxes[0][1])
                    yoloe_bbox[2] = math.ceil(yoloe_bboxes[0][2])
                    yoloe_bbox[3] = math.ceil(yoloe_bboxes[0][3])

                    image = load_image(os.path.join(args.image_folder, temp_image_name))
                    prompt = "Please only provide the bounding box coordinate (as a list) of the region \"" + temp_character + "\" describes. Do not include any JSON formatting or additional text in the response."
                    output_text = pipe((prompt, image))
                    output_text = output_text.text

                    try:
                        output_text = output_text.replace("json", "").replace("```", "")
                        llm_bbox = eval(output_text)
                        temp_iou, area1, area2 = iou_cal(yoloe_bbox, llm_bbox)
                    except:
                        valid_character_list_bbox[temp_character] = None
                        continue

                    
                    if temp_iou > 0.7:
                        if area1 > area2:
                            valid_character_list_bbox[temp_character] = yoloe_bbox
                        else:
                            valid_character_list_bbox[temp_character] = llm_bbox
                    else:
                        try:
                            yoloe_bbox_crop_img = now_image.crop(yoloe_bbox)
                            llm_bbox_crop_img = now_image.crop(llm_bbox)
                            image_pair = [yoloe_bbox_crop_img, llm_bbox_crop_img]
                            blip_process_input = blip_processor(image_pair, [temp_character["main_character"]], return_tensors="pt", padding=True).to(blip_model.device)
                        except:
                            valid_character_list_bbox[temp_character] = None
                            continue
                        cosine_score = blip_model(pixel_values=blip_process_input['pixel_values'], input_ids=blip_process_input['input_ids'], attention_mask=blip_process_input['attention_mask'], use_itm_head=False).itm_score
                        
                        # if cosine_score[0] < 0.4 and cosine_score[1] < 0.4:
                        #     valid_character_list_bbox[temp_character] = None
                        #     continue
                        
                        if cosine_score[0] > cosine_score[1]:
                            valid_character_list_bbox[temp_character] = yoloe_bbox
                        else:
                            valid_character_list_bbox[temp_character] = llm_bbox

                # character locations
                valid_character_temp_character_locations = {}
                temp_character_locations = temp_piece["character_locations"]
                temp_character_locations_dict = {}
                for temp_character_locations_piece in temp_character_locations:
                    temp_character_locations_dict[temp_character_locations_piece["main_character"]] = temp_character_locations_piece["position"]

                for temp_character in valid_character_list_bbox:
                    if not temp_character in temp_character_locations_dict:
                        continue
                    valid_character_temp_character_locations[temp_character] = 0
                    
                    if valid_character_list_bbox[temp_character] == None:
                        continue

                    bbox_img = now_image.copy()
                    draw = ImageDraw.Draw(bbox_img)
                    try:
                        draw.rectangle(valid_character_list_bbox[temp_character], outline="red", width=5)
                    except:
                        continue

                    norm_bbox = [0,0,0,0]
                    norm_bbox[0] = round(valid_character_list_bbox[temp_character][0]/image_size, 2)
                    norm_bbox[1] = round(valid_character_list_bbox[temp_character][1]/image_size, 2)
                    norm_bbox[2] = round(valid_character_list_bbox[temp_character][2]/image_size, 2)
                    norm_bbox[3] = round(valid_character_list_bbox[temp_character][3]/image_size, 2)
                    prompt = "Analyze whether the character \"" + temp_character + "\" (marked with a red bounding box at coordinates " + str(norm_bbox) + ") is located in " + temp_character_locations_dict[temp_character].lower() + ". Please only respond with 'yes' or 'no'."
                    output_text = pipe((prompt, bbox_img))
                    output_text = output_text.text
                    
                    if 'yes' in output_text.lower():
                        valid_character_temp_character_locations[temp_character] = 1

                # character locations stastic
                temp_image_state["character_location_all"] = 0
                temp_image_state["character_location_success"] = 0

                character_locations_all_count += len(temp_character_locations_dict)
                temp_image_state["character_location_all"] += len(temp_character_locations_dict)

                for temp_character in valid_character_temp_character_locations:
                    character_locations_success_count += valid_character_temp_character_locations[temp_character]
                    temp_image_state["character_location_success"] += valid_character_temp_character_locations[temp_character]

                new_keep_image_state.append(temp_image_state)
                print(character_locations_all_count)
                print(character_locations_success_count)


            elif temp_piece["mini_bench_target"] in ["background", "light", "style"]:
                # scene attributes
                temp_image_state["scene_attributes_all"] = {"background":0, "light":0, "style":0, "spatial":0}
                temp_image_state["scene_attributes_success"] = {"background":0, "light":0, "style":0, "spatial":0}
                for temp_scene_attribute in temp_piece["scene_attributes"]:
                    if temp_scene_attribute["scene_attribute"] == temp_piece["mini_bench_target"]:
                        image = load_image(os.path.join(args.image_folder, temp_image_name))
                        prompt = "Analyze whether the " + temp_scene_attribute["scene_attribute"] + " condition in this image match the following description: \"" + temp_scene_attribute["content"] +"\". Please only respond with 'yes' or 'no'."
                        output_text = pipe((prompt, image))
                        output_text = output_text.text

                        scene_attrbutes_all_count[temp_scene_attribute["scene_attribute"]] += 1
                        temp_image_state["scene_attributes_all"][temp_scene_attribute["scene_attribute"]] += 1
                        
                        if 'yes' in output_text.lower():
                            scene_attrbutes_success_count[temp_scene_attribute["scene_attribute"]] += 1
                            temp_image_state["scene_attributes_success"][temp_scene_attribute["scene_attribute"]] += 1
                
                new_keep_image_state.append(temp_image_state)
                print(scene_attrbutes_all_count)
                print(scene_attrbutes_success_count)


            elif temp_piece["mini_bench_target"] == "entity_relationship":
                # object presence
                valid_character_list_bbox = {}
                for temp_character in valid_character_list:
                    names = [temp_character]
                    yoloe_model.set_classes(names, yoloe_model.get_text_pe(names))
                    try:
                        results = yoloe_model.predict(os.path.join(args.image_folder, temp_image_name), verbose=False)
                    except:
                        Image.open(os.path.join(args.image_folder, temp_image_name)).convert("RGB").save(os.path.join(args.image_folder, temp_image_name))
                        results = yoloe_model.predict(os.path.join(args.image_folder, temp_image_name), verbose=False)
                    yoloe_bboxes = results[0].boxes.xyxy.tolist()

                    if len(yoloe_bboxes) == 0:
                        valid_character_list_bbox[temp_character] = None
                        continue
                    yoloe_bbox = [0,0,0,0]
                    yoloe_bbox[0] = int(yoloe_bboxes[0][0])
                    yoloe_bbox[1] = int(yoloe_bboxes[0][1])
                    yoloe_bbox[2] = math.ceil(yoloe_bboxes[0][2])
                    yoloe_bbox[3] = math.ceil(yoloe_bboxes[0][3])

                    image = load_image(os.path.join(args.image_folder, temp_image_name))
                    prompt = "Please only provide the bounding box coordinate (as a list) of the region \"" + temp_character + "\" describes. Do not include any JSON formatting or additional text in the response."
                    output_text = pipe((prompt, image))
                    output_text = output_text.text

                    try:
                        output_text = output_text.replace("json", "").replace("```", "")
                        llm_bbox = eval(output_text)
                        temp_iou, area1, area2 = iou_cal(yoloe_bbox, llm_bbox)
                    except:
                        valid_character_list_bbox[temp_character] = None
                        continue

                    if temp_iou > 0.7:
                        if area1 > area2:
                            valid_character_list_bbox[temp_character] = yoloe_bbox
                        else:
                            valid_character_list_bbox[temp_character] = llm_bbox
                    else:
                        try:
                            yoloe_bbox_crop_img = now_image.crop(yoloe_bbox)
                            llm_bbox_crop_img = now_image.crop(llm_bbox)
                            image_pair = [yoloe_bbox_crop_img, llm_bbox_crop_img]
                            blip_process_input = blip_processor(image_pair, [temp_character["main_character"]], return_tensors="pt", padding=True).to(blip_model.device)
                        except:
                            valid_character_list_bbox[temp_character] = None
                            continue
                        cosine_score = blip_model(pixel_values=blip_process_input['pixel_values'], input_ids=blip_process_input['input_ids'], attention_mask=blip_process_input['attention_mask'], use_itm_head=False).itm_score
                        
                        if cosine_score[0] > cosine_score[1]:
                            valid_character_list_bbox[temp_character] = yoloe_bbox
                        else:
                            valid_character_list_bbox[temp_character] = llm_bbox


                # scene attributes
                temp_image_state["scene_attributes_all"] = {"background":0, "light":0, "style":0, "spatial":0}
                temp_image_state["scene_attributes_success"] = {"background":0, "light":0, "style":0, "spatial":0}
                for temp_scene_attribute in temp_piece["scene_attributes"]:
                    if temp_scene_attribute["scene_attribute"] == "spatial":
                        for temp_spatial in temp_scene_attribute["content"]:
                            bbox_img = now_image.copy()
                            draw = ImageDraw.Draw(bbox_img)
                            add_character_str = ""
                            for temp_contain_character_id, temp_contain_character in enumerate(valid_character_list_bbox):
                                if valid_character_list_bbox[temp_contain_character] == None:
                                    continue

                                try:
                                    draw.rectangle(valid_character_list_bbox[temp_contain_character], outline="red", width=5)
                                except:
                                    continue

                                if not add_character_str == "":
                                    add_character_str += ", "
                                add_character_str += "\"" + temp_contain_character + "\""
                                
                            
                            if add_character_str == "":
                                output_text = "no"
                            else:
                                prompt = "The provided image contains only characters: " + add_character_str + " (highlighted with red bounding boxes). Analyze whether the spatial condition in this image match the following description: \"" + temp_spatial +"\". Please only respond with 'yes' or 'no'."
                                output_text = pipe((prompt, bbox_img))
                                output_text = output_text.text

                            scene_attrbutes_all_count["spatial"] += 1
                            temp_image_state["scene_attributes_all"]["spatial"] += 1

                            if 'yes' in output_text.lower():
                                scene_attrbutes_success_count["spatial"] += 1
                                temp_image_state["scene_attributes_success"]["spatial"] += 1
                new_keep_image_state.append(temp_image_state)
                print(scene_attrbutes_all_count)
                print(scene_attrbutes_success_count)
                

    overall_json = {}
    overall_json["character_attributes_all_count"] = character_attributes_all_count
    overall_json["character_attributes_success_count"] = character_attributes_success_count
    overall_json["character_attributes_detect_count"] = character_attributes_detect_count
    overall_json["character_locations_all_count"] = character_locations_all_count
    overall_json["character_locations_success_count"] = character_locations_success_count
    overall_json["scene_attrbutes_all_count"] = scene_attrbutes_all_count
    overall_json["scene_attrbutes_success_count"] = scene_attrbutes_success_count
    with open(os.path.join(args.output_log_dir, args.output_name_prefix + "_overall_eval.json"), "a") as f:
        json.dump(overall_json, f)


    with open(os.path.join(args.output_log_dir, args.output_name_prefix + "_no_found_image.json"), "a") as f:
        json.dump(no_found_image_id, f)

    with open(os.path.join(args.output_log_dir, args.output_name_prefix + "_detail_score.json"), "a") as f:
        json.dump(new_keep_image_state, f)