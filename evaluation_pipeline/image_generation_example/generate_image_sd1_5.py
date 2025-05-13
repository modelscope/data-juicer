import torch
from diffusers import StableDiffusionPipeline
import json
import tqdm
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="stable-diffusion-v1-5")
    parser.add_argument('--prompt_path', type=str, default="./data_json.json")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument('--image_output_dir', type=str, default="./output_image/")
    args=parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda")
    new_data = []

    with open(args.prompt_path, "r") as f:
        data = json.load(f)

    valid_image_count = 0
    for temp_piece in tqdm.tqdm(data):
        try:
            prompt = temp_piece["polished_prompt"]
            image = pipe(
                prompt,
                height=512,
                width=512,
                num_inference_steps=50,
            ).images[0]
        
            image_name = temp_piece["dataset_target"] + "_SD1_5_" + str(valid_image_count) + "_" + str(temp_piece["image_id"])
            image.save(os.path.join(args.image_output_dir, image_name))

            temp_json = {}
            temp_json["output_image_name"] = image_name
            temp_json["image_id"] = temp_piece["dataset_target"] + "_" + temp_piece["image_id"]
            new_data.append(temp_json)

            valid_image_count += 1

        except:
            continue

    with open(args.output_json, "a") as f:
        json.dump(new_data, f)
    