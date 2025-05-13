from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import argparse
import os
import json
import tqdm
import torch


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
    with open(args.data_json_path, "r") as f:
        file = json.load(f)
        for data in tqdm.tqdm(file):

            try: 
                new_json_line = {}

                # background
                image = load_image(os.path.join(args.image_folder, data['image_id']))
                prompt = "I will provide you with an image and its corresponding description. The description of the corresponding image is: \"" + data['ori_caption'] + "\" You need to tell me the background of the image. Please only reply with one or two concise sentences related to the \'background\'."
                output_text = pipe((prompt, image))
                output_text = output_text.text

                new_json_line["background"] = output_text


                # light
                image = load_image(os.path.join(args.image_folder, data['image_id']))
                prompt = "I will provide you with an image and its corresponding description. The description of the corresponding image is: \"" + data['ori_caption'] + "\" You need to analyze the lighting conditions in the image. This may relate to time (e.g., morning, noon, dusk), light intensity (e.g., bright, dim, harsh, soft), indoor/outdoor setting, or light source positioning (e.g., front-lit, backlit, side-lit). Please respond only with one or two concise lighting-related sentences."
                output_text = pipe((prompt, image))
                output_text = output_text.text

                new_json_line["light"] = output_text

                # style
                image = load_image(os.path.join(args.image_folder, data['image_id']))
                prompt = "I will provide you with an image and its corresponding description. The description of the corresponding image is: \"" + data['ori_caption'] + "\" You need to identify the style of the image, such as realistic photo, comic, artwork, vintage photo, etc. Please reply concisely with only one or two sentences related to \'style\'. Do not describe the image content."
                output_text = pipe((prompt, image))
                output_text = output_text.text

                new_json_line["style"] = output_text


                # spatial relationship
                image = load_image(os.path.join(args.image_folder, data['image_id']))
                prompt = "I will provide you with an image and its corresponding description, as well as a list of the main characters in the image. The description of the corresponding image is: \"" + data['ori_caption'] + "\" The main character list is: \"" + str(data["main_character"]) + "\" You need to analyze the spatial relationships between the main characters. Please only reply with sentences related to \'spatial relationships.\' Please respond in the form of a string list."
                output_text = pipe((prompt, image))
                output_text = output_text.text
            except:
                continue

            try:
                output_text = eval(output_text)
                new_json_line["spatial"] = output_text
            except:
                new_json_line["spatial"] = output_text

            print(new_json_line)

            new_json_line["main_character"] = data["main_character"]
            new_json_line["image_id"] = data["image_id"]
            new_json_line["dataset_target"] = data["dataset_target"]
            new_json_line["ori_caption"] = data["ori_caption"]
            new_json.append(new_json_line)


    with open(args.output_json, "a") as f:
        json.dump(new_json, f)