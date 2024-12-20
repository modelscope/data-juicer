import sys
sys.path.append("./LLaVA/")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import json
import argparse
import tqdm
import torch
import os
from transformers import CLIPImageProcessor, CLIPModel, AutoTokenizer
from transformers import BlipProcessor, BlipForImageTextRetrieval
import random
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp2p_bbox_json_path', type=str, default="./filtered_inp2p_bbox.json")
    parser.add_argument('--llava_path', type=str, default="./llava-v1.6-vicuna-7b")
    parser.add_argument('--clip_path', type=str, default="./clip-vit-base-patch32")
    parser.add_argument('--blip_path', type=str, default="./blip-itm-large-coco")
    parser.add_argument('--img_dir', type=str, default="./instructpix2pix")
    parser.add_argument('--output_img_dir', type=str, default="./new_edit_img")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--qa_turns', type=int, default=6)
    parser.add_argument("--output_file", type=str, default=".")
    args=parser.parse_args()

    return args

def get_sub_list(ori_list, indice_list):
    new_list = []
    for i in indice_list:
        new_list.append(ori_list[i])
    return new_list


class InferenceDataset(Dataset):

    def __init__(self, args):
        with open(args.inp2p_bbox_json_path, "r") as f:
            self.json_file = json.load(f)
        self.args = args

    def __len__(self) -> int:
        return len(self.json_file)

    @torch.no_grad()
    def __getitem__(self, idx: int):


        return self.json_file[idx]["path"].replace("./prompt-to-prompt-with-sdxl/output", "/"), self.json_file[idx]["bbox"]


# Adopted from https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/blob/main/llava-multi-images.py
def concatenate_images_horizontal(images, bar_width):
    # calc total width of imgs + dist between them
    total_width = sum(img.width for img in images) + bar_width * (len(images) - 1)
    # calc max height from imgs
    height = max(img.height for img in images)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    # init var to track current width pos
    current_width = 0
    for img in images:
        # paste img in new_img at current width
        new_img.paste(img, (current_width, 0))
        # update current width for next img
        current_width += img.width + bar_width

    return new_img

def iou_filter(now_bbox, bbox_list, thresh):
    for temp in bbox_list:
        x1_overlap = max(now_bbox[0], temp[0])
        y1_overlap = max(now_bbox[1], temp[1])
        x2_overlap = min(now_bbox[2], temp[2])
        y2_overlap = min(now_bbox[3], temp[3])

        w = max(0, x2_overlap - x1_overlap)
        h = max(0, y2_overlap - y1_overlap)
        overlap_area = w * h

        iou = overlap_area / ((now_bbox[2] - now_bbox[0])*(now_bbox[3]-now_bbox[1]) + (temp[2]-temp[0])*(temp[3]-temp[1]) - overlap_area)

        if iou > thresh:
            return True

    return False




if __name__ == "__main__":
    
    args = parse_args()
    device = args.device


    blip_processor = BlipProcessor.from_pretrained(args.blip_path)
    blip_model = BlipForImageTextRetrieval.from_pretrained(args.blip_path, torch_dtype=torch.float16).to(device).half()


    clip_model = CLIPModel.from_pretrained(args.clip_path).to(device).half()
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_path)
    clip_tokenizer = AutoTokenizer.from_pretrained(args.clip_path)

    llava_path = args.llava_path
    model_path = os.path.expanduser(llava_path)
    model_base = None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, use_flash_attn=True, load_4bit=False)
    # model = model.to(device)

    batch_size = 1
    dataset = InferenceDataset(args)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False)



    count = 0
    new_json = []

    for image_path, filtered_targets in tqdm.tqdm(dataloader):

        image_path = args.img_dir + image_path[0][1:]

        image1 = Image.open(image_path + "_0.jpg").convert('RGB')
        image2 = Image.open(image_path + "_1.jpg").convert('RGB')
        image2 = image2.resize((image1.size[0], image1.size[1]))
        image_tensor1 = process_images([image1], image_processor, model.config)[0]
        image_tensor2 = process_images([image2], image_processor, model.config)[0]

        image_tensor_list = []
        prompt_list = []
        concat_image_tensor_list = []
        bbox_list = []
        crop_bbox_list = []
        red_bbox_img_list1 = []
        red_bbox_img_list2 = []


        if len(filtered_targets) > 0 :
            
            prompt_bbox = "Please provide a clear description for this region: " #
            for temp_idx, temp_target in enumerate(filtered_targets):
                if (temp_target[2] - temp_target[0]) * (temp_target[3] - temp_target[1]) < (image1.size[0] * image1.size[1]) / 400:
                    continue 

                if len(concat_image_tensor_list) == args.qa_turns:
                    break

                temp_image1 = image1.copy()
                temp_image2 = image2.copy()
                draw1 = ImageDraw.ImageDraw(temp_image1)
                draw2 = ImageDraw.ImageDraw(temp_image2)

                extend_width = 5
                if temp_target[0] - extend_width >= 0:
                    extend_x1 = temp_target[0] - extend_width
                else:
                    extend_x1 = 0
                 
                if temp_target[1] - extend_width >= 0:
                    extend_y1 = temp_target[1] - extend_width
                else:
                    extend_y1 = 0

                if temp_target[2] + extend_width <= image1.size[0]:
                    extend_x2 = temp_target[2] + extend_width
                else:
                    extend_x2 = image1.size[0]
                
                if temp_target[3] +extend_width <= image1.size[1]:
                    extend_y2 = temp_target[3] +extend_width
                else:
                    extend_y2 = image1.size[1]

                crop_bbox_list.append((int(extend_x1), int(extend_y1), int(extend_x2), int(extend_y2)))

                draw1.rectangle(((extend_x1, extend_y1),(extend_x2, extend_y2)), fill=None, outline='red', width=3)
                draw2.rectangle(((extend_x1, extend_y1),(extend_x2, extend_y2)), fill=None, outline='red', width=3)
                red_bbox_img_list1.append(temp_image1)
                red_bbox_img_list2.append(temp_image2)
                concat_image = concatenate_images_horizontal([temp_image1, temp_image2], 20)
                concat_image.save("./label_img/" + str(temp_idx) + "concat_image_pil.jpg")
                concat_image_tensor = process_images([concat_image], image_processor, model.config)[0]
                

                image_tensor_list.append(image_tensor1) # img1
                image_tensor_list.append(image_tensor2) # img2
                concat_image_tensor_list.append(concat_image_tensor)

                temp_bbox_x1 = str(round(float(temp_target[0] / image1.size[0]), 2))
                temp_bbox_y1 = str(round(float(temp_target[1] / image1.size[1]), 2))
                temp_bbox_x2 = str(round(float(temp_target[2] / image1.size[0]), 2))
                temp_bbox_y2 = str(round(float(temp_target[3] / image1.size[1]), 2))



                while(len(temp_bbox_x1) < 4):
                    temp_bbox_x1 = temp_bbox_x1 + "0"
                while(len(temp_bbox_y1) < 4):
                    temp_bbox_y1 = temp_bbox_y1 + "0"
                while(len(temp_bbox_x2) < 4):
                    temp_bbox_x2 = temp_bbox_x2 + "0"
                while(len(temp_bbox_y2) < 4):
                    temp_bbox_y2 = temp_bbox_y2 + "0"

                str_bbox = "[" + temp_bbox_x1 + ", " + temp_bbox_y1 + ", "+ temp_bbox_x2 + ", " + temp_bbox_y2 +"]"
                prompt = prompt_bbox + str_bbox + "."
                bbox_list.append(str_bbox)
                # prompt = "Analyse the the left image and the right image (separated by the black vertical bar). What differences are present within the red-bordered areas of the two images? The box may potentially be empty. Answer these questions in a concise sentence."
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                conv = conv_templates["vicuna_v1"].copy()
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                input_ids = input_ids.to(device=model.device, non_blocking=True)

                prompt_list.append(input_ids) # img1
                prompt_list.append(input_ids) # img2



            if len(prompt_list) == 0:
                continue

            input_ids = torch.stack(prompt_list).to(model.device)
            image_tensor = torch.stack(image_tensor_list).to(dtype=torch.float16, device=model.device, non_blocking=True)
            # concat_image_tensor = torch.stack(concat_image_tensor_list).to(dtype=torch.float16, device=model.device, non_blocking=True)

            temperature = 0
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image1.size] * len(image_tensor_list),
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=64,
                    use_cache=True) #                     

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            captions_outputs = outputs

            # similar captions filter
            filter_caption_idx = []
            caption_tokens = clip_tokenizer(outputs, padding=True, return_tensors="pt").to("cuda")
            caption_text_features = clip_model.get_text_features(**caption_tokens)
            # print(caption_text_features.shape)
            for temp_idx in range(int(len(outputs)/2)):
                cos = torch.cosine_similarity(caption_text_features[2 * temp_idx], caption_text_features[2 * temp_idx + 1], dim=0)
                if cos<0.85: #0.9
                    filter_caption_idx.append(temp_idx)

            if len(filter_caption_idx) == 0:
                continue

            # caption quality filter
            final_filter_idx = []
            filter_captions = []
            filter_idx = []
            crop_img_list = []
            for temp_idx in filter_caption_idx:
                crop_pil_img1 = image1.crop(crop_bbox_list[temp_idx])
                crop_pil_img2 = image2.crop(crop_bbox_list[temp_idx])
                crop_img_list.append(crop_pil_img1)
                crop_img_list.append(crop_pil_img2)
                filter_captions.append(outputs[temp_idx * 2])
                filter_captions.append(outputs[temp_idx * 2 + 1])
            
            # print(crop_inputs_list)
            crop_inputs_list = blip_processor(crop_img_list, filter_captions, return_tensors="pt", padding=True).to("cuda", torch.float16)
            cosine_score = blip_model(**crop_inputs_list, use_itm_head=False).itm_score
            
            for temp_idx in range(len(filter_caption_idx)):
                if cosine_score[temp_idx * 2][temp_idx * 2] > 0.35 and cosine_score[temp_idx * 2 + 1][temp_idx * 2 + 1] > 0.35: #0.3
                    final_filter_idx.append(filter_caption_idx[temp_idx])

            if len(final_filter_idx) == 0:
                continue

            red_bbox_img_list1 = get_sub_list(red_bbox_img_list1, final_filter_idx)
            red_bbox_img_list2 = get_sub_list(red_bbox_img_list2, final_filter_idx)
            concat_image_tensor_list = get_sub_list(concat_image_tensor_list, final_filter_idx)
            concat_image_tensor = torch.stack(concat_image_tensor_list).to(dtype=torch.float16, device=model.device, non_blocking=True)

            concat_image_input_ids_list = []
            for temp_idx in final_filter_idx:
                prompt = "Analyse the the left image and the right image (separated by the black vertical bar). The detail within the red bounding box in the left image is: " + outputs[temp_idx * 2] + ", " + \
                "while the detail within the red bounding box in the right image is: " + outputs[temp_idx * 2 + 1] + ". What is their difference? Answer with a few concise sentences."#
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                conv = conv_templates["vicuna_v1"].copy()
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                input_ids = input_ids.to(device=model.device, non_blocking=True)

                concat_image_input_ids_list.append(input_ids)

            
            max_len = 0
            for temp_idx in range(len(concat_image_input_ids_list)):
                max_len = max(max_len, len(concat_image_input_ids_list[temp_idx]))

            for temp_idx in range(len(concat_image_input_ids_list)):
                if len(concat_image_input_ids_list[temp_idx]) < max_len:
                    concat_image_input_ids_list[temp_idx] = torch.cat((torch.zeros(max_len-len(concat_image_input_ids_list[temp_idx])).to(model.device), concat_image_input_ids_list[temp_idx])).long()

            concat_image_input_ids = torch.stack(concat_image_input_ids_list)

            temperature = 0
            with torch.inference_mode():
                output_ids = model.generate(
                    concat_image_input_ids.to(model.device),
                    images=concat_image_tensor,
                    image_sizes=[concat_image.size] * len(concat_image_tensor_list),
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True) #                     

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            
            

            for enumerate_idx, temp_idx in enumerate(final_filter_idx):
                temp_json = {}
                
                temp_json["conversations"] = []
                temp_json["bbox"] = bbox_list[temp_idx] 

                temp_json["captions1"] = captions_outputs[temp_idx * 2]
                temp_json["captions2"] = captions_outputs[temp_idx * 2 + 1]

                temp_conversation = {}
                temp_conversation["from"] = "human"
                temp_conversation["value"] = "Analyse the the left image and the right image (separated by the black vertical bar). What is the difference between the red bounding box area in each image? Answer the question in a few concise sentences."
                temp_json["conversations"].append(temp_conversation)

                temp_conversation = {}
                temp_conversation["from"] = "gpt"
                temp_conversation["value"] = outputs[enumerate_idx]
                temp_json["conversations"].append(temp_conversation)

                temp_json["path"] = os.path.join(args.output_img_dir, image_path.split("/")[-1] + "_" + str(enumerate_idx))
                red_bbox_img_list1[enumerate_idx].save(temp_json["path"] + "_0.jpg")
                red_bbox_img_list2[enumerate_idx].save(temp_json["path"] + "_1.jpg")

                new_json.append(temp_json)

            

    with open(os.path.join(args.output_file), "w") as new_f:
        new_f.write(json.dumps(new_json))

