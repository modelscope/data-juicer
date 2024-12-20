import sys
sys.path.append("../object_replacement/LLaVA/")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from ultralytics import FastSAM, YOLO
from transformers import CLIPImageProcessor, CLIPModel
import json
import numpy as np
import tqdm
import cv2
from PIL import Image, ImageDraw, ImageColor
import torch
from torch.utils.data import Dataset, DataLoader
import difflib
from transformers import BlipProcessor, BlipForImageTextRetrieval
import re
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import random
import argparse
import os
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import word_tokenize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default="./inpaint")
    parser.add_argument('--vit_path', type=str, default="clip-vit-base-patch32")
    parser.add_argument('--blip_path', type=str, default="blip-itm-large-coco")
    parser.add_argument('--fastsam_path', type=str, default="FastSAM-x.pt")
    parser.add_argument('--json_path', type=str, default="./filtered_file_new_edit_09_098_3.json")
    parser.add_argument('--sd_model_path', type=str, default="stable-diffusion-xl-base-1.0")
    parser.add_argument('--split_name', type=str, default="0")
    parser.add_argument("--output_file", type=str, default="inpaint_0.json")
    parser.add_argument('--mllm_path', type=str, default="./llava-v1.6-vicuna-7b")
    args=parser.parse_args()

    return args


def iou_filter(samples, iou_thresh):
    x1 = samples[:, 0]
    y1 = samples[:, 1]
    x2 = samples[:, 2]
    y2 = samples[:, 3]
    scores = samples[:, 4]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep_boxes = []
    index = scores.argsort() # Ascending

    while len(index) > 0:
        i = index[0]
        keep_boxes.append(i)

        x1_overlap = np.maximum(x1[i], x1[index[1:]])
        y1_overlap = np.maximum(y1[i], y1[index[1:]])
        x2_overlap = np.minimum(x2[i], x2[index[1:]])
        y2_overlap = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x2_overlap - x1_overlap + 1)
        h = np.maximum(0, y2_overlap - y1_overlap + 1)
        overlap_area = w * h

        ious = overlap_area / (areas[i] + areas[index[1:]] - overlap_area)

        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1] # update

    return samples[keep_boxes]


class InferenceDataset_for_FastSAM(Dataset):

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.image_path = json.load(f)

    def __len__(self) -> int:
        return len(self.image_path)

    @torch.no_grad()
    def __getitem__(self, idx: int):
        # image_array1 = cv2.cvtColor(cv2.imread(self.image_path[idx] + "_0.jpg"), cv2.COLOR_BGR2RGB)
        # image_array2 = cv2.cvtColor(cv2.imread(self.image_path[idx] + "_1.jpg"), cv2.COLOR_BGR2RGB)
        # image_array2 = cv2.resize(image_array2,(image_array1.shape[1],image_array1.shape[0]))

        return self.image_path[idx]



class InferenceDataset_for_clip(Dataset):

    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self) -> int:
        return len(self.image_list)

    @torch.no_grad()
    def __getitem__(self, idx: int):

        return self.image_list[idx]


class InferenceDataset_for_blip(Dataset):

    def __init__(self, pixel_values):
        self.pixel_values = pixel_values

    def __len__(self) -> int:
        return len(self.pixel_values)

    @torch.no_grad()
    def __getitem__(self, idx: int):

        return self.pixel_values[idx]


def is_noun(word):

    pos_tagged = pos_tag([word])
    pos = pos_tagged[0][1]

    return pos in ['NN', 'NNS', 'NNP', 'NNPS']


def compare_text_index(text1, text2):
    # matcher = difflib.SequenceMatcher(a=text1, b=text2)
    # diff_report = matcher.get_opcodes()

    # for tag, i1, i2, j1, j2 in diff_report:
    #     if tag == 'replace':
    #         return text1[i1:i2], text2[j1:j2]
        

    d = difflib.Differ()
    diff = d.compare(re.sub(r'[^\w\s]', '', text1.lower().replace(" ", "\n")).splitlines(), re.sub(r'[^\w\s]', '', text2.lower().replace(" ", "\n")).splitlines())

    text1 = ""
    text2 = ""

    for line in diff:
        if line.startswith('+'):
            if not is_noun(line.replace("+ ", "")):
                continue
            text1 = text1 + " " + line.replace("+ ", "")
        elif line.startswith('-'):
            if not is_noun(line.replace("- ", "")):
                continue
            text2 = text2 + " " + line.replace("- ", "")


    return text1.strip(), text2.strip()




if __name__ == "__main__":
    args = parse_args()

    llava_path = args.mllm_path
    model_path = os.path.expanduser(llava_path)
    model_base = None
    model_name = get_model_name_from_path(model_path)
    tokenizer, mllm, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, use_flash_attn=False, load_4bit=False)

    choice = ["A", "B"]
    rand_choice = [0, 1]
    new_json = []
    device = "cuda"

    vision_model = CLIPModel.from_pretrained(args.vit_path).to(device).half()
    processor = CLIPImageProcessor.from_pretrained(args.vit_path)

    blip_processor = BlipProcessor.from_pretrained(args.blip_path)
    blip_model = BlipForImageTextRetrieval.from_pretrained(args.blip_path, torch_dtype=torch.float16).to(device).half()

    fastSAM_model = FastSAM(args.fastsam_path)

    image_dataset = InferenceDataset_for_FastSAM(args.json_path)
    print(args.json_path)
    dataloader_fastsam = DataLoader(image_dataset, batch_size=16, drop_last=False)

    pipeline = AutoPipelineForInpainting.from_pretrained(args.sd_model_path, torch_dtype=torch.float16).to("cuda")

    with torch.no_grad():
        for image_path_list in tqdm.tqdm(dataloader_fastsam):
            # print(len(image_path_list))
            image_list1 = []
            image_list2 = []
            for temp_idx_list in range(len(image_path_list["path"])):
                image_array1 = cv2.cvtColor(cv2.imread(image_path_list["path"][temp_idx_list].replace("./", "../new_edit_data/") + "_0.jpg"), cv2.COLOR_BGR2RGB)
                image_array1 = cv2.resize(image_array1, (512, 512))
                image_list1.append(image_array1)

                image_array2 = cv2.cvtColor(cv2.imread(image_path_list["path"][temp_idx_list].replace("./", "../new_edit_data/") + "_1.jpg"), cv2.COLOR_BGR2RGB)
                image_array2 = cv2.resize(image_array2,(512, 512))
                image_list2.append(image_array2)

            masks1 = fastSAM_model(image_list1, retina_masks=True, imgsz=1024, conf=0.1, iou=0.5, verbose=False)
            masks2 = fastSAM_model(image_list2, retina_masks=True, imgsz=1024, conf=0.1, iou=0.5, verbose=False)

            for temp_idx_mask in range(len(image_path_list["path"])):

                # print(image_path_list["input"][temp_idx_mask])
                # print(image_path_list["output"][temp_idx_mask])

                noun1, noun2 = compare_text_index(image_path_list["input"][temp_idx_mask], image_path_list["output"][temp_idx_mask])
                if noun1 == "" or noun2 == "":
                    continue 

                # print(image_path_list["input"][temp_idx_mask])
                # print(image_path_list["output"][temp_idx_mask])
                # print(noun1)
                # print(noun2)

                temp_mask1 = masks1[temp_idx_mask]
                temp_mask2 = masks2[temp_idx_mask]
                if len(temp_mask1.boxes.xyxy) == 0:
                    continue

                image_array1 = image_list1[temp_idx_mask]
                image_array2 = image_list2[temp_idx_mask]

                image_targets = []
                image_targets_pos = []
                diff_targets_1 = []
                diff_targets_2 = []
                mask_list = []

                with torch.no_grad():
                    for temp_target, temp_ori_mask in zip(temp_mask1.boxes.xyxy, temp_mask1.masks):
                        # crop_img = image_array1[int(temp_target['bbox'][1]):int(temp_target['bbox'][1])+int(temp_target['bbox'][3]),int(temp_target['bbox'][0]):int(temp_target['bbox'][0])+int(temp_target['bbox'][2]),:]
                        crop_img = image_array1[int(temp_target[1]):int(temp_target[3]),int(temp_target[0]):int(temp_target[2]),:]
                        img = Image.fromarray(crop_img)
                        image_targets.append(img)
                        # image1_targets_pos.append(temp_target['bbox'])
                        image_targets_pos.append(temp_target.cpu())

                        cv2_img = np.where(temp_ori_mask.data.cpu().numpy().transpose((1,0,2)).transpose((0,2,1)) > 0.5, 255, 0).astype(np.uint8)
                        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
                        pil_img = Image.fromarray(cv2_img)
                        mask_list.append(pil_img)

                        crop_img_same_pos = image_array2[int(temp_target[1]):int(temp_target[3]),int(temp_target[0]):int(temp_target[2]),:]
                        img = Image.fromarray(crop_img_same_pos)
                        image_targets.append(img)

                    num_image1_targets = len(image_targets)

                    # cv2.imwrite(os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img0.jpg"), image_array1)
                    # cv2.imwrite(os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img1.jpg"), image_array2)

                    for temp_target, temp_ori_mask in zip(temp_mask2.boxes.xyxy, temp_mask2.masks):
                        # crop_img = image_array1[int(temp_target['bbox'][1]):int(temp_target['bbox'][1])+int(temp_target['bbox'][3]),int(temp_target['bbox'][0]):int(temp_target['bbox'][0])+int(temp_target['bbox'][2]),:]
                        crop_img = image_array2[int(temp_target[1]):int(temp_target[3]),int(temp_target[0]):int(temp_target[2]),:]
                        img = Image.fromarray(crop_img)
                        image_targets.append(img)
                        # image1_targets_pos.append(temp_target['bbox'])
                        image_targets_pos.append(temp_target.cpu())

                        cv2_img = np.where(temp_ori_mask.data.cpu().numpy().transpose((1,0,2)).transpose((0,2,1)) > 0.5, 255, 0).astype(np.uint8)
                        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
                        pil_img = Image.fromarray(cv2_img)
                        mask_list.append(pil_img)

                        crop_img_same_pos = image_array1[int(temp_target[1]):int(temp_target[3]),int(temp_target[0]):int(temp_target[2]),:]
                        img = Image.fromarray(crop_img_same_pos)
                        image_targets.append(img)
                
                # print(len(image_targets))
                
                if len(image_targets) == 0:
                    continue
                try:
                    image_targets_clip = processor(image_targets, return_tensors="pt")['pixel_values'].half()
                except: 
                    continue
                image_dataset = InferenceDataset_for_clip(image_targets_clip)
                dataloader_clip = DataLoader(image_dataset, batch_size=256, drop_last=False)
                
                image_feature = None
                for batch in dataloader_clip:
                    temp_image_feature = vision_model.get_image_features(batch.to(vision_model.device))
                    if image_feature == None:
                        image_feature = temp_image_feature.to(torch.float32)
                    else:
                        image_feature = torch.cat((image_feature, temp_image_feature.to(torch.float32)), dim = 0)


                try:
                    image_targets_blip = blip_processor(image_targets, [noun1, noun2], return_tensors="pt", padding=True).to(device, torch.float16)
                except: 
                    continue
                
                input_ids = image_targets_blip['input_ids']
                attention_mask = image_targets_blip['attention_mask']
                image_dataset = InferenceDataset_for_blip(image_targets_blip['pixel_values'])
                dataloader_blip = DataLoader(image_dataset, batch_size=256, drop_last=False)
                blip_itm_score = None
                for pixel_values in dataloader_blip:
                    cosine_score = blip_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, use_itm_head=False).itm_score
                    if blip_itm_score == None:
                        blip_itm_score = cosine_score.to(torch.float32)
                    else:
                        blip_itm_score = torch.cat((blip_itm_score, cosine_score.to(torch.float32)), dim = 0)
                    
                # print(blip_itm_score)

                for temp_idx_cos in range(0, image_feature.shape[0], 2):
                    
                    # thresh = 0.3
                    # if blip_itm_score[temp_idx_cos][0] < thresh and blip_itm_score[temp_idx_cos + 1][1] < thresh:
                    #     continue



                    cos = torch.cosine_similarity(image_feature[temp_idx_cos], image_feature[temp_idx_cos + 1], dim=0)
                    if (cos<0.9):
                        temp_diff_target = []
                        temp_diff_target.extend(image_targets_pos[int(temp_idx_cos/2)])
                        temp_diff_target.append(cos.cpu())
                        
                        if temp_idx_cos < num_image1_targets:
                            temp_diff_target.append(1)
                        else:
                            temp_diff_target.append(2)

                        temp_diff_target.append(int(temp_idx_cos/2))

                        if temp_idx_cos < num_image1_targets:
                            diff_targets_1.append(temp_diff_target)
                        else:
                            diff_targets_2.append(temp_diff_target)
                        

                # print(len(diff_targets))

                if len(diff_targets_1) + len(diff_targets_2) == 0:
                    continue

                if len(diff_targets_1) > 0 :
                    filtered_targets_1 = iou_filter(np.array(diff_targets_1), 0.5)
                
                    for temp_idx, temp_filtered_target in enumerate(filtered_targets_1):

                        if temp_idx == 3:
                            break

                        bbox_x1 = temp_filtered_target[0]
                        bbox_y1 = temp_filtered_target[1]
                        bbox_x2 = temp_filtered_target[2]
                        bbox_y2 = temp_filtered_target[3]

                        which_img = temp_filtered_target[5]
                        mask = mask_list[int(temp_filtered_target[6])]

                        if which_img == 1:

                            # image 1
                            prompt = "background, nothing, 8k"
                            new_image = pipeline(prompt=prompt, image=Image.fromarray(image_array1), mask_image=mask, strength=0.85, guidance_scale=0, num_inference_steps=4).images[0]
                            temp_image_array1 = Image.fromarray(image_array1)

                            # mllm captioning
                            prompt_bbox = f"Please provide a clear description for this region: [{str(bbox_x1)}, {str(bbox_y1)}, {str(bbox_x2)}, {str(bbox_y2)}]."
                            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt_bbox
                            conv = conv_templates["vicuna_v1"].copy()
                            conv.append_message(conv.roles[0], prompt)
                            conv.append_message(conv.roles[1], None)
                            prompt = conv.get_prompt()
                            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                            input_ids = input_ids.to(device=mllm.device, non_blocking=True)
                            image_tensor = process_images([temp_image_array1], image_processor, mllm.config)[0]

                            temperature = 0
                            with torch.inference_mode():
                                output_ids = mllm.generate(
                                    input_ids.unsqueeze(0),
                                    images=image_tensor.to(dtype=torch.float16, device=mllm.device, non_blocking=True).unsqueeze(0),
                                    image_sizes=[temp_image_array1.size],
                                    do_sample=True if temperature > 0 else False,
                                    temperature=temperature,
                                    top_p=None,
                                    num_beams=1,
                                    max_new_tokens=64,
                                    use_cache=True) #   
                            caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

                            # print(caption)

                            # caption quality filter
                            crop_pil_img1 = temp_image_array1.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
                            crop_pil_img2 = new_image.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
                            crop_inputs_list = blip_processor([crop_pil_img1, crop_pil_img2], [caption], return_tensors="pt", padding=True).to("cuda", torch.float16)
                            cosine_score = blip_model(**crop_inputs_list, use_itm_head=False).itm_score
                            if cosine_score[0][0] < 0.35 or cosine_score[1][0] > 0.2:
                                continue


                            now_choice = random.choice(rand_choice)
                            draw = ImageDraw.ImageDraw(temp_image_array1)
                            draw.rectangle(((bbox_x1-15, bbox_y1-15),(bbox_x2+15, bbox_y2+15)), fill=None, outline='red', width=3)
                            temp_image_array1.save(os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img0_" + str(temp_idx) + "_" + args.split_name + "_" + str(1-now_choice) + ".jpg"))
                            # cv2.imwrite(os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img0_" + args.split_name + "_" + str(1-now_choice) + ".jpg"), image_array1)
                            
                            draw = ImageDraw.ImageDraw(new_image)
                            draw.rectangle(((bbox_x1-15, bbox_y1-15),(bbox_x2+15, bbox_y2+15)), fill=None, outline='red', width=3)
                            new_image.save(os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img0_" + str(temp_idx) + "_" + args.split_name + "_" + str(now_choice) + ".jpg"))

                            temp_json = {}
                            temp_json["bbox"] = [int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)]
                            temp_json["conversations"] = []

                            temp_conversation = {}
                            temp_conversation["from"] = "human"
                            temp_conversation["value"] = f"Analyse the the left image and the right image (separated by the black vertical bar). Which image has the object related to \"{caption}\" within the red bounding box?\nA. the left image\nB. the right image\nAnswer with the option's letter from the given choices directly."
                            temp_json["conversations"].append(temp_conversation)

                            temp_conversation = {}
                            temp_conversation["from"] = "gpt"
                            temp_conversation["value"] = choice[now_choice]
                            temp_json["conversations"].append(temp_conversation)

                            temp_json["path"] = os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img0_" + str(temp_idx) + "_" + args.split_name)

                            new_json.append(temp_json)

                if len(diff_targets_2) > 0:
                    filtered_targets_2 = iou_filter(np.array(diff_targets_2), 0.5)

                    for temp_idx, temp_filtered_target in enumerate(filtered_targets_2):

                        if temp_idx == 3:
                            break

                        bbox_x1 = temp_filtered_target[0]
                        bbox_y1 = temp_filtered_target[1]
                        bbox_x2 = temp_filtered_target[2]
                        bbox_y2 = temp_filtered_target[3]

                        which_img = temp_filtered_target[5]
                        mask = mask_list[int(temp_filtered_target[6])]

                        if which_img == 2:

                            # image 2
                            prompt = "background, nothing, 8k"
                            # print(noun2)
                            # print(prompt)
                            new_image = pipeline(prompt=prompt, image=Image.fromarray(image_array2), mask_image=mask, strength=0.85, guidance_scale=0, num_inference_steps=4).images[0]
                            temp_image_array2 = Image.fromarray(image_array2)

                            # mllm captioning
                            prompt_bbox = f"Please provide a clear description for this region: [{str(bbox_x1)}, {str(bbox_y1)}, {str(bbox_x2)}, {str(bbox_y2)}]."
                            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt_bbox
                            conv = conv_templates["vicuna_v1"].copy()
                            conv.append_message(conv.roles[0], prompt)
                            conv.append_message(conv.roles[1], None)
                            prompt = conv.get_prompt()
                            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                            input_ids = input_ids.to(device=mllm.device, non_blocking=True)
                            image_tensor = process_images([temp_image_array2], image_processor, mllm.config)[0]

                            temperature = 0
                            with torch.inference_mode():
                                output_ids = mllm.generate(
                                    input_ids.unsqueeze(0),
                                    images=image_tensor.to(dtype=torch.float16, device=mllm.device, non_blocking=True).unsqueeze(0),
                                    image_sizes=[temp_image_array2.size],
                                    do_sample=True if temperature > 0 else False,
                                    temperature=temperature,
                                    top_p=None,
                                    num_beams=1,
                                    max_new_tokens=64,
                                    use_cache=True) #   
                            caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

                            # print(caption)

                            # caption quality filter
                            crop_pil_img1 = temp_image_array2.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
                            crop_pil_img2 = new_image.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
                            crop_inputs_list = blip_processor([crop_pil_img1, crop_pil_img2], [caption], return_tensors="pt", padding=True).to("cuda", torch.float16)
                            cosine_score = blip_model(**crop_inputs_list, use_itm_head=False).itm_score
                            # print(cosine_score)
                            if cosine_score[0][0] < 0.35 or cosine_score[1][0] > 0.2:
                                continue


                            now_choice = random.choice(rand_choice)
                            draw = ImageDraw.ImageDraw(temp_image_array2)
                            draw.rectangle(((bbox_x1-15, bbox_y1-15),(bbox_x2+15, bbox_y2+15)), fill=None, outline='red', width=3)
                            temp_image_array2.save(os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img1_" + str(temp_idx) + "_" + args.split_name + "_" + str(1-now_choice) + ".jpg"))
                            # cv2.imwrite(os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img1_" + args.split_name + "_" + str(1-now_choice) + ".jpg"), image_array2)
                            
                            draw = ImageDraw.ImageDraw(new_image)
                            draw.rectangle(((bbox_x1-15, bbox_y1-15),(bbox_x2+15, bbox_y2+15)), fill=None, outline='red', width=3)
                            new_image.save(os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img1_" + str(temp_idx) + "_" + args.split_name + "_" + str(now_choice) + ".jpg"))

                            temp_json = {}
                            temp_json["bbox"] = [int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)]
                            temp_json["conversations"] = []

                            temp_conversation = {}
                            temp_conversation["from"] = "human"
                            temp_conversation["value"] = f"Analyse the the left image and the right image (separated by the black vertical bar). Which image has the object related to \"{caption}\" within the red bounding box?\nA. the left image\nB. the right image\nAnswer with the option's letter from the given choices directly."
                            temp_json["conversations"].append(temp_conversation)

                            temp_conversation = {}
                            temp_conversation["from"] = "gpt"
                            temp_conversation["value"] = choice[now_choice]
                            temp_json["conversations"].append(temp_conversation)

                            temp_json["path"] = os.path.join(args.out_path, image_path_list["path"][temp_idx_mask].split("/")[-1] + "_img1_" + str(temp_idx) + "_" + args.split_name)

                            new_json.append(temp_json)
            #     break
            # break
    
    print(len(new_json))
    # print(new_json)
    with open(args.output_file, "w") as f:
        f.write(json.dumps(new_json))
