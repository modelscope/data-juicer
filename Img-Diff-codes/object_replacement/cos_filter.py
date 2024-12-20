import os
from transformers import CLIPImageProcessor, CLIPModel
import json
import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import torch
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default="./prompt-to-prompt-with-sdxl/output")
    parser.add_argument('--json_path', type=str, default="./gen_0.json")
    parser.add_argument('--split_name', type=str, default="0")
    parser.add_argument('--clip_vit_path', type=str, default="clip-vit-base-patch32")
    args=parser.parse_args()

    return args


class InferenceDataset(Dataset):

    def __init__(self, images_path, json_path, processor, split_name):
        self.images_path = images_path
        with open(json_path, "r") as f:
            self.json_file = json.load(f)
        self.idx_list = []
        for idx, i in tqdm.tqdm(enumerate(self.json_file)):
            if os.path.exists(os.path.join(self.images_path, split_name) + "_" + str(idx) + "_0.jpg") and "---" not in i["output"] and "replaced" not in i["output"].lower():
                self.idx_list.append(idx)
        print(len(self.idx_list))
        self.processor = processor
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.idx_list)

    @torch.no_grad()
    def __getitem__(self, idx: int):
        image_path1 = os.path.join(self.images_path, self.split_name) + "_" + str(self.idx_list[idx]) + "_0.jpg"
        image1 = Image.open(image_path1).convert('RGB')

        image_path2 = os.path.join(self.images_path, self.split_name) + "_" + str(self.idx_list[idx]) + "_1.jpg"
        image2 = Image.open(image_path2).convert('RGB')

        images_tensor = self.processor([image1, image2], return_tensors="pt")['pixel_values']

        return images_tensor[0], images_tensor[1], os.path.join(self.images_path, self.split_name) + "_" + str(self.idx_list[idx]), self.json_file[self.idx_list[idx]]


if __name__ == "__main__":
    args = parse_args()

    device = "cuda"
    filtered_file = []
    images_path = args.folder_path
    json_path = args.json_path

    vision_model = CLIPModel.from_pretrained(args.clip_vit_path).to(device).half()
    processor = CLIPImageProcessor.from_pretrained(args.clip_vit_path)

    image_dataset = InferenceDataset(images_path, json_path, processor, args.split_name)
    sampler = SequentialSampler(image_dataset)
    dataloader = DataLoader(image_dataset, sampler=sampler, batch_size=2, drop_last=False)

    with torch.no_grad():
        for (image1_batch, image2_batch, images_path, json_piece) in tqdm.tqdm(dataloader):
            image1_batch_feature = vision_model.get_image_features(image1_batch.to(vision_model.device))
            image2_batch_feature = vision_model.get_image_features(image2_batch.to(vision_model.device))

            cos_list = torch.cosine_similarity(image1_batch_feature, image2_batch_feature, dim=1)
            for temp_idx in range(image1_batch_feature.shape[0]):
                cos = cos_list[temp_idx]
                temp_json = {}
                temp_json["path"] = images_path[temp_idx]
                temp_json["cos"] = round(float(cos), 3)
                temp_json["input"] = json_piece["input"][temp_idx]
                temp_json["output"] = json_piece["output"][temp_idx]
                with open(f"./filtered_file_new_caption_{args.split_name}.txt", "a") as txt_f:
                    txt_f.write(str(temp_json) + "\n")

    
    