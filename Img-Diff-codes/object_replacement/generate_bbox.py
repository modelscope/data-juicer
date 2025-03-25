from ultralytics import FastSAM, YOLO
from transformers import CLIPImageProcessor, CLIPModel
import json
import numpy as np
import tqdm
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import difflib
from transformers import BlipProcessor, BlipForImageTextRetrieval
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import argprase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_path', type=str, default="clip-vit-base-patch32")
    parser.add_argument('--blip_path', type=str, default="blip-itm-large-coco")
    parser.add_argument('--fastsam_path', type=str, default="FastSAM-x.pt")
    parser.add_argument('--json_path', type=str, default="filtered_file_new_edit_09_098_3.json")
    parser.add_argument('--output_file', type=str, default="bbox_file_3.json")
    
    args=parser.parse_args()

    return args

def is_noun(word):
    # print(word)
    pos_tagged = pos_tag([word])
    pos = pos_tagged[0][1]

    if not pos in ['NN', 'NNS', 'NNP', 'NNPS']:
        return False
    
    return True


def is_adj(word):
    # print(word)
    pos_tagged = pos_tag([word])
    pos = pos_tagged[0][1]

    if not pos in ["JJ", "JJR", "JJS"]:
        return False
    
    return True



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


def compare_text_index(text1, text2):

    text1_split = []
    text2_split = []

    lemmatizer=WordNetLemmatizer()

    d = difflib.Differ()
    diff = d.compare(re.sub(r'[^\w\s]', '', text1.lower().replace(" ", "\n")).splitlines(), re.sub(r'[^\w\s]', '', text2.lower().replace(" ", "\n")).splitlines())


    for line in diff:
        if line.startswith('+'):
            text2_split.append(lemmatizer.lemmatize(line.replace("+ ", "")))
        elif line.startswith('-'):
            text1_split.append(lemmatizer.lemmatize(line.replace("- ", "")))

    text1 = []
    text2 = []

    for temp_idx, temp_word1 in enumerate(text1_split):
        if temp_word1 not in text2_split:
            if is_noun(temp_word1):
                text1.append(temp_word1)

    for temp_idx, temp_word2 in enumerate(text2_split):
        if temp_word2 not in text1_split:
            if is_noun(temp_word2):
                text2.append(temp_word2)

    return text1, text2





if __name__ == "__main__":
    args = parse_args()

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

    with torch.no_grad():
        for image_path_list in tqdm.tqdm(dataloader_fastsam):
            # print(len(image_path_list))
            image_list1 = []
            image_list2 = []
            for temp_idx_list in range(len(image_path_list["path"])):
                image_array1 = cv2.cvtColor(cv2.imread(image_path_list["path"][temp_idx_list] + "_0.jpg"), cv2.COLOR_BGR2RGB)
                image_list1.append(image_array1)

                image_array2 = cv2.cvtColor(cv2.imread(image_path_list["path"][temp_idx_list] + "_1.jpg"), cv2.COLOR_BGR2RGB)
                image_array2 = cv2.resize(image_array2,(image_array1.shape[1],image_array1.shape[0]))
                image_list2.append(image_array2)

            masks1 = fastSAM_model(image_list1, retina_masks=True, imgsz=1024, conf=0.05, iou=0.5, verbose=False)
            masks2 = fastSAM_model(image_list2, retina_masks=True, imgsz=1024, conf=0.05, iou=0.5, verbose=False)

            for temp_idx_mask in range(len(image_path_list["path"])):

                # print(image_path_list["input"][temp_idx_mask])
                # print(image_path_list["output"][temp_idx_mask])

                noun1, noun2 = compare_text_index(image_path_list["input"][temp_idx_mask], image_path_list["output"][temp_idx_mask])
                if noun1 == [] and noun2 == []:
                    continue 

                # print(noun1)
                # print(noun2)

                temp_mask1 = masks1[temp_idx_mask]
                temp_mask2 = masks2[temp_idx_mask]
                if len(temp_mask1.boxes.xyxy) + len(temp_mask2.boxes.xyxy) == 0:
                    continue

                image_array1 = image_list1[temp_idx_mask]
                image_array2 = image_list2[temp_idx_mask]

                image_targets = []
                image_targets_pos = []
                diff_targets = []

                with torch.no_grad():
                    for temp_target in temp_mask1.boxes.xyxy:
                        # crop_img = image_array1[int(temp_target['bbox'][1]):int(temp_target['bbox'][1])+int(temp_target['bbox'][3]),int(temp_target['bbox'][0]):int(temp_target['bbox'][0])+int(temp_target['bbox'][2]),:]
                        crop_img = image_array1[int(temp_target[1]):int(temp_target[3]),int(temp_target[0]):int(temp_target[2]),:]
                        img = Image.fromarray(crop_img)
                        image_targets.append(img)
                        # image1_targets_pos.append(temp_target['bbox'])
                        image_targets_pos.append(temp_target.cpu())

                        crop_img_same_pos = image_array2[int(temp_target[1]):int(temp_target[3]),int(temp_target[0]):int(temp_target[2]),:]
                        img = Image.fromarray(crop_img_same_pos)
                        image_targets.append(img)

                    num_image1_targets = len(image_targets)

                    


                    for temp_target in temp_mask2.boxes.xyxy:
                        # crop_img = image_array1[int(temp_target['bbox'][1]):int(temp_target['bbox'][1])+int(temp_target['bbox'][3]),int(temp_target['bbox'][0]):int(temp_target['bbox'][0])+int(temp_target['bbox'][2]),:]
                        crop_img = image_array2[int(temp_target[1]):int(temp_target[3]),int(temp_target[0]):int(temp_target[2]),:]
                        img = Image.fromarray(crop_img)
                        image_targets.append(img)
                        # image1_targets_pos.append(temp_target['bbox'])
                        image_targets_pos.append(temp_target.cpu())

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


                temp_noun = []
                temp_noun.extend(noun1)
                temp_noun.extend(noun2)
                try:
                    image_targets_blip = blip_processor(image_targets, temp_noun, return_tensors="pt", padding=True).to(device, torch.float16)
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
                    
                    thresh = 0.35 # 0.35
                    # if blip_itm_score[temp_idx_cos][0] < thresh and blip_itm_score[temp_idx_cos + 1][1] < thresh: # and -> or
                    #     continue

                    not_match = True # effective object

                    for temp_count in range(len(noun1)):
                        if blip_itm_score[temp_idx_cos][temp_count] > thresh:
                            not_match = False
                            break

                    if not_match:
                        for temp_count in range(len(noun2)):
                            if blip_itm_score[temp_idx_cos + 1][len(noun1) + temp_count] > thresh:
                                not_match = False
                                break

                    if not_match:
                        continue


                    cos = torch.cosine_similarity(image_feature[temp_idx_cos], image_feature[temp_idx_cos + 1], dim=0)
                    if (cos<0.85): # 0.95 0.85
                        temp_diff_target = []
                        temp_diff_target.extend(image_targets_pos[int(temp_idx_cos/2)])
                        temp_diff_target.append(cos.cpu())
                        
                        if temp_idx_cos < num_image1_targets:
                            temp_diff_target.append(1)
                        else:
                            temp_diff_target.append(2)
                        diff_targets.append(temp_diff_target)

                # print(len(diff_targets))

                if len(diff_targets) == 0:
                    continue


                filtered_targets = iou_filter(np.array(diff_targets), 0.5)

                temp_new_json = {}
                temp_new_json["path"] = image_path_list["path"][temp_idx_mask]
                temp_filtered_bbox = []
                for temp_idx_bbox, temp_filtered_targets in enumerate(filtered_targets):
                    if temp_idx_bbox == 10:
                        break
                    temp_bbox_num = []
                    for num in temp_filtered_targets[0:4]:
                        temp_bbox_num.append(round(float(num), 1))
                    temp_filtered_bbox.append(temp_bbox_num)
                temp_new_json["bbox"] = temp_filtered_bbox

                # print(len(temp_filtered_bbox))
                new_json.append(temp_new_json)

                if len(new_json) % 1000 == 0:
                    print(len(new_json))

            # break
    
    print(len(new_json))
    with open(args.output_file, "w") as f:
        f.write(json.dumps(new_json))
