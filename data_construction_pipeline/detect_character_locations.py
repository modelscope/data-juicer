from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from transformers import BlipProcessor, BlipForImageTextRetrieval
import argparse
import os
import json
import tqdm
from ultralytics import YOLOE
from PIL import Image
import random
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen2_5_vl_model_path', type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument('--yoloe_model_path', type=str, default="yoloe-11l-seg.pt")
    parser.add_argument('--blip_model_path', type=str, default="Salesforce/blip-itm-large-flickr")
    parser.add_argument('--data_json_path', type=str, default="./data_json.json")
    parser.add_argument('--image_folder', type=str, default="./image_folder")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--gpu_nums', type=int, default=1)

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
    pipe = pipeline(args.qwen2_5_vl_model_path, backend_config=TurbomindEngineConfig(tp=args.gpu_nums))


    new_json = []

    with open(args.data_json_path, "r") as f:
        data = json.load(f)
    for temp_line in tqdm.tqdm(data):

        try:
            names = temp_line["main_character"]
        except:
            continue
        yoloe_model.set_classes(names, yoloe_model.get_text_pe(names))

        # Execute prediction for specified categories on an image
        results = yoloe_model.predict(os.path.join(args.image_folder, temp_line["image_id"]), verbose=False)
        yoloe_bboxes = results[0].boxes.xyxy.tolist()
        bboxes_cls = results[0].boxes.cls.tolist()

        valid_main_character = []
        seen = []
        for temp_bbox_idx in range(len(yoloe_bboxes)):
            if bboxes_cls[temp_bbox_idx] in seen:
                continue
            seen.append(bboxes_cls[temp_bbox_idx])
            temp_bbox_json = {}
            temp_bbox_json["main_character"] = names[int(bboxes_cls[temp_bbox_idx])]
            temp_bbox_json["yoloe_bbox"] = [round(yoloe_bboxes[temp_bbox_idx][0]), round(yoloe_bboxes[temp_bbox_idx][1]), round(yoloe_bboxes[temp_bbox_idx][2]), round(yoloe_bboxes[temp_bbox_idx][3])]
            valid_main_character.append(temp_bbox_json)


        final_bboxes = []
        for temp_character in valid_main_character:

            image = load_image(os.path.join(args.image_folder, temp_line["image_id"]))
            prompt = "Please only provide the bounding box coordinate of the region \"" + temp_character["main_character"] + "\" describes."
            output_text = pipe((prompt, image))
            output_text = output_text.text

            

            try:
                output_text = output_text.replace("json", "").replace("```", "")
                output_text = eval(output_text)
                temp_character["llm_bbox"] = output_text[0]["bbox_2d"]
                final_bboxes.append(temp_character)
            except:
                continue
        

        now_image = Image.open(os.path.join(args.image_folder, temp_line["image_id"]))
        final_filterd_character = []
        for temp_character_idx, temp_character in enumerate(final_bboxes):
            temp_iou, area1, area2 = iou_cal(temp_character["yoloe_bbox"], temp_character["llm_bbox"])

            if temp_iou > 0.7:
                if area1 > area2:
                    temp_json = {}
                    temp_json["main_character"] = temp_character["main_character"]
                    temp_json["bbox"] = temp_character["yoloe_bbox"]
                    final_filterd_character.append(temp_json)
                else:
                    temp_json = {}
                    temp_json["main_character"] = temp_character["main_character"]
                    temp_json["bbox"] = temp_character["llm_bbox"]
                    final_filterd_character.append(temp_json)
            else:
                yoloe_bbox_crop_img = now_image.crop(temp_character["yoloe_bbox"])
                llm_bbox_crop_img = now_image.crop(temp_character["llm_bbox"])

                image_pair = [yoloe_bbox_crop_img, llm_bbox_crop_img]
                try:
                    blip_process_input = blip_processor(image_pair, [temp_character["main_character"]], return_tensors="pt", padding=True).to(blip_model.device)
                except:
                    continue
                cosine_score = blip_model(pixel_values=blip_process_input['pixel_values'], input_ids=blip_process_input['input_ids'], attention_mask=blip_process_input['attention_mask'], use_itm_head=False).itm_score
                
                if cosine_score[0] < 0.4 and cosine_score[1] < 0.4:
                    continue
                
                if cosine_score[0] > cosine_score[1]:
                    temp_json = {}
                    temp_json["main_character"] = temp_character["main_character"]
                    temp_json["bbox"] = temp_character["yoloe_bbox"]
                    final_filterd_character.append(temp_json)
                else:
                    temp_json = {}
                    temp_json["main_character"] = temp_character["main_character"]
                    temp_json["bbox"] = temp_character["llm_bbox"]
                    final_filterd_character.append(temp_json)


        print(final_filterd_character)

        if len(final_filterd_character) == 0:
            continue

        temp_json = {}
        temp_json["image_id"] = temp_line["image_id"]
        temp_json["dataset_target"] = temp_line["dataset_target"]
        temp_json["main_character_list"] = final_filterd_character
        temp_json["ori_caption"] = temp_line["ori_caption"]
        temp_json["all_main_character"] = temp_line["main_character"]
        new_json.append(temp_json)


    with open(args.output_json, "a") as f:
        json.dump(new_json, f)