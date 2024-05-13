import os

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def save_videos_grid(videos: torch.Tensor,
                     path: str,
                     rescale=False,
                     n_rows=6,
                     fps=12,
                     imageio_backend=True):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1 / fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path,
                        format='GIF',
                        append_images=outputs,
                        save_all=True,
                        duration=100,
                        loop=0)
