import argparse
import os
import json
import tqdm
from PIL import Image
import random


upper_area = [0.3, 0, 0.7, 0.5] # The upper part of the image
lower_area = [0.3, 0.5, 0.7, 1] # The lower part of the image
left_area = [0, 0.3, 0.5, 0.7] # The left part of the image
right_area = [0.5, 0.3, 1, 0.7] # The right part of the image
middle_area = [0.3, 0.3, 0.7, 0.7] # The middle part of the image
upper_left_area = [0, 0, 0.4, 0.4] # The upper left part of the image
lower_left_area = [0, 0.6, 0.4, 1] # The lower left part of the image
upper_right_area = [0.6, 0, 1, 0.4] # The upper right part of the image
lower_right_area = [0.6, 0.6, 1, 1] # The lower right part of the image

area_list = [upper_area, lower_area, left_area, right_area, middle_area,
            upper_left_area, lower_left_area, upper_right_area, lower_right_area]
id_to_area = ["The upper part of the image", "The lower part of the image", "The left part of the image",
            "The right part of the image", "The middle part of the image", "The upper left part of the image",
            "The lower left part of the image", "The upper right part of the image", "The lower right part of the image"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--character_bbox_data_json_path', type=str, default="./data_json.json")
    parser.add_argument('--image_folder', type=str, default="./image_folder")
    parser.add_argument('--output_json', type=str, default="./output.json")
    args=parser.parse_args()

    return args


def occupy_cal(bbox1, bbox2):

    min_x1 = min(bbox1[0], bbox2[0])
    max_x1 = max(bbox1[0], bbox2[0])
    min_y1 = min(bbox1[1], bbox2[1])
    max_y1 = max(bbox1[1], bbox2[1])

    max_x2 = max(bbox1[2], bbox2[2])
    min_x2 = min(bbox1[2], bbox2[2])
    max_y2 = max(bbox1[3], bbox2[3])
    min_y2 = min(bbox1[3], bbox2[3])

    if min_x2 - max_x1 < 0 or min_y2 - max_y1 < 0:
        return 0

    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    intersection_area = (min_x2-max_x1) * (min_y2-max_y1)
    occupy = intersection_area / area1

    return occupy


if __name__ == "__main__":
    args = parse_args()

    with open(args.character_bbox_data_json_path, "r") as f:
        data = json.load(f)
        new_json = []

        for temp_line in data:
            temp_img = Image.open(os.path.join(args.image_folder, temp_line["image_id"]))

            new_character_list = []
            for temp_character in temp_line["main_character_list"]:
                norm_bbox = []
                norm_bbox.append(temp_character["bbox"][0] / temp_img.size[0])
                norm_bbox.append(temp_character["bbox"][1] / temp_img.size[1])
                norm_bbox.append(temp_character["bbox"][2] / temp_img.size[0])
                norm_bbox.append(temp_character["bbox"][3] / temp_img.size[1])

                temp_iou = []
                for temp_area in area_list:
                    temp_iou.append(occupy_cal(norm_bbox, temp_area))

                temp_iou_dict = {}
                for temp_area_name_id, temp_area_name in enumerate(id_to_area):
                    temp_iou_dict[temp_area_name] = temp_iou[temp_area_name_id]
                temp_iou = sorted(temp_iou_dict.items(), key=lambda x:x[1], reverse=True)

                if temp_iou[0][1] > 0.75:
                    temp_character["position"] = temp_iou[0][0]
                else:
                    temp_character["position"] = None

                new_character_list.append(temp_character)
            
            temp_line["main_character_list"] = new_character_list
            new_json.append(temp_line)

    with open(args.output_json, "a") as f:
        json.dump(new_json, f)
