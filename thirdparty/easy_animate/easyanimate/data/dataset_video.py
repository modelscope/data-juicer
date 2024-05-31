import csv
import io
import json
import math
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from torch.utils.data.dataset import Dataset

VIDEO_READER_TIMEOUT = 10

def get_random_mask(shape):
    f, c, h, w = shape

    mask_index = np.random.randint(0, 4)
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)
    if mask_index == 0:
        mask[1:, :, :, :] = 1
    elif mask_index == 1:
        mask_frame_index = 1
        mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
    elif mask_index == 2:
        center_x = torch.randint(0, w, (1, )).item()
        center_y = torch.randint(0, h, (1, )).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3,
                                     (1, )).item()  # 方块的宽度范围
        block_size_y = torch.randint(h // 4, h // 4 * 3,
                                     (1, )).item()  # 方块的高度范围

        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)
        mask[:, :, start_y:end_y, start_x:end_x] = 1
    elif mask_index == 3:
        center_x = torch.randint(0, w, (1, )).item()
        center_y = torch.randint(0, h, (1, )).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3,
                                     (1, )).item()  # 方块的宽度范围
        block_size_y = torch.randint(h // 4, h // 4 * 3,
                                     (1, )).item()  # 方块的高度范围

        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)

        mask_frame_before = np.random.randint(0, f // 2)
        mask_frame_after = np.random.randint(f // 2, f)
        mask[mask_frame_before:mask_frame_after, :, start_y:end_y,
             start_x:end_x] = 1
    else:
        raise ValueError(f"The mask_index {mask_index} is not define")
    return mask


def get_video_reader_batch(video_reader, batch_index):
    return video_reader.get_batch(batch_index).asnumpy()


class VideoDataset(Dataset):

    def __init__(self,
                 json_path,
                 video_folder=None,
                 sample_size=256,
                 sample_stride=4,
                 sample_n_frames=16,
                 enable_bucket=False,
                 enable_inpaint=False):
        print(f"loading annotations from {json_path} ...")

        if json_path.lower().endswith('jsonl'):
            with open(json_path, 'r') as fin:
                self.dataset = [json.loads(line.strip()) for line in fin]
        else:
            self.dataset = json.load(open(json_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.enable_bucket = enable_bucket
        self.enable_inpaint = enable_inpaint

        sample_size = tuple(sample_size) if not isinstance(
            sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5],
                                 inplace=True),
        ])

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_id, name = video_dict['videos'], video_dict['text']

        if isinstance(video_id, list):
            video_id = video_id[0]

        video_dir = video_id
        if not os.path.exists(video_id):
            if self.video_folder:
                video_dir = os.path.join(self.video_folder, video_id)
            else:
                raise ValueError(
                    f"{video_id} does not exist, please change it to absolute path or ser video_folder"
                )

        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        clip_length = min(video_length,
                          (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx,
                                  start_idx + clip_length - 1,
                                  self.sample_n_frames,
                                  dtype=int)

        try:
            sample_args = (video_reader, batch_index)
            pixel_values = func_timeout(
                VIDEO_READER_TIMEOUT,
                get_video_reader_batch,
                args=sample_args
            )
        except FunctionTimedOut:
            raise ValueError(f"Read {idx} timeout.")
        except Exception as e:
            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

        if not self.enable_bucket:
            pixel_values = torch.from_numpy(pixel_values).permute(
                    0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader

        # remove special token
        name = name.replace('<__dj__video>', '').replace('<|__dj__eoc|>',
                                                         '').strip()

        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                print("Error info:", e)
                idx = random.randint(0, self.length - 1)

        if not self.enable_bucket:
            pixel_values = self.pixel_transforms(pixel_values)
        if self.enable_inpaint:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (
                1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample = dict(pixel_values=pixel_values,
                          mask_pixel_values=mask_pixel_values,
                          mask=mask,
                          text=name)
        else:
            sample = dict(pixel_values=pixel_values, text=name)
        return sample
