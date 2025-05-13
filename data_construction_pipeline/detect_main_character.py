from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import argparse
import os
import json
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen2_5_vl_model_path', type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument('--data_json_path', type=str, default="./data_json.json")
    parser.add_argument('--image_folder', type=str, default="./image_folder")
    parser.add_argument('--output_json', type=str, default="./output.json")
    parser.add_argument("--dataset_target",type=str ,choices= ["docci" , "ln_coco", "ln_flickr"] , required=True)
    parser.add_argument('--gpu_nums', type=int, default=1)
    args=parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    pipe = pipeline(args.qwen2_5_vl_model_path, backend_config=TurbomindEngineConfig(tp=args.gpu_nums))

    new_json = []
    temp_idx = 0
    with open(args.data_json_path, "r") as f:
        file = json.load(f)
        for data in tqdm.tqdm(file):
            
            if args.dataset_target == "docci":
                image_id = data['image_file'].replace(".jpg", "")
                if not os.path.exists(os.path.join(args.image_folder, data['image_file'])):
                    continue

                ori_caption = data["description"]
                image = load_image(os.path.join(args.image_folder, data['image_file']))
                prompt = "I will provide you with an image and its corresponding description. You need to identify and count the main characters in the image (e.g., key people, key animals, key objects). The output should only be in JSON format, including the number of main characters and a list of their descriptions, as shown in the example: {\"count\": 3, \"main_character\": [\"man in a blue shirt\", \"black cat\", \"skateboard\"]}. Below, I will provide the description of the corresponding image: \"" + data["description"] + "\" Please identify the main characters and output the result in JSON format."
                output_text = pipe((prompt, image))
                output_text = output_text.text



            elif args.dataset_target == "ln_flickr":
                image_id = data['image_id']
                if not os.path.exists(os.path.join(args.image_folder, data['image_id'] + ".jpg")):
                    continue

                ori_caption = data['caption']
                image = load_image(os.path.join(args.image_folder, data['image_id'] + ".jpg"))
                prompt = "I will provide you with an image and its corresponding description. You need to identify and count the main characters in the image (e.g., key people, key animals, key objects). The output should only be in JSON format, including the number of main characters and a list of their descriptions, as shown in the example: {\"count\": 3, \"main_character\": [\"man in a blue shirt\", \"black cat\", \"skateboard\"]}. Below, I will provide the description of the corresponding image: \"" + data['caption'] + "\" Please identify the main characters and output the result in JSON format."
                output_text = pipe((prompt, image))
                output_text = output_text.text


            elif args.dataset_target == "ln_coco":
                image_id = data['image_id']
                if len(image_id) < 12:
                    image_id = "0" * (12 - len(image_id)) + image_id
                if not os.path.exists(os.path.join(args.image_folder, image_id + ".jpg")):
                    continue

                ori_caption = data['caption']
                image = load_image(os.path.join(args.image_folder, image_id + ".jpg"))
                prompt = "I will provide you with an image and its corresponding description. You need to identify and count the main characters in the image (e.g., key people, key animals, key objects). The output should only be in JSON format, including the number of main characters and a list of their descriptions, as shown in the example: {\"count\": 3, \"main_character\": [\"man in a blue shirt\", \"black cat\", \"skateboard\"]}. Below, I will provide the description of the corresponding image: \"" + data['caption'] + "\" Please identify the main characters and output the result in JSON format."
                output_text = pipe((prompt, image))
                output_text = output_text.text


            try:
                result_json = "{" + output_text.replace("\n", "").replace("\\", "").split("{")[-1].split("}")[-2] + "}"
                print(result_json)
                
                json_str = eval(result_json)
            except:
                print("broken_json")
                print("now_len: " + str(len(new_json)))
                continue


            json_str["image_id"] = image_id + ".jpg"
            json_str["dataset_target"] = args.dataset_target
            json_str["ori_caption"] = ori_caption
            new_json.append(json_str)


    with open(args.output_json, "a") as f:
         json.dump(new_json, f)