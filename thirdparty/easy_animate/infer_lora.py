import argparse
import json
import os
import sys

import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer3d import Transformer3DModel
from easyanimate.pipeline.pipeline_easyanimate import EasyAnimatePipeline
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import save_videos_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")

    parser.add_argument(
        "--prompt_info_path",
        type=str,
        default=None,
        help=("The prompts to produce videos."),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help=("The size of generated image. It can be 256 or 512."),
    )
    parser.add_argument(
        "--chunks_num",
        type=int,
        default=1,
        help=("The number of prompts divided for different devices."),
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help=("The chunk_id in current device."),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=("The batch size in each inferance."),
    )
    parser.add_argument(
        "--video_num_per_prompt",
        type=int,
        default=3,
        help=("The number of generated videos for each prompt."),
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=("The config of the model in inferance."),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="DPM++",
        choices=['Euler', 'Euler A', 'DPM++', 'PNDM', 'DDIM'],
        help=
        "Choose the sampler in 'Euler' 'Euler A' 'DPM++' 'PNDM' and 'DDIM'",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=
        ("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
         " 1.10.and an Nvidia Ampere GPU."),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=
        ("If you want to load the weight from other transformers, input its path."
         ),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=(
            "If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help=("The path to the trained lora weight."),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=("The save path of generated videos."),
    )

    args = parser.parse_args()

    return args


def get_chunk(prompt_dicts, chunk_id, chunks_num):
    l = len(prompt_dicts)
    chunk_len = ((l - 1) // chunks_num) + 1
    f = chunk_id * chunk_len
    t = (chunk_id + 1) * chunk_len
    return prompt_dicts[f:t]


def get_batch(arr, batch_size):
    l = len(arr)
    batch_num = ((l - 1) // batch_size) + 1
    batch_arr = []
    for i in range(batch_num):
        batch_arr.append(arr[i * batch_size:(i + 1) * batch_size])
    return batch_arr


def main():
    args = parse_args()

    video_length = 16
    fps = 12
    guidance_scale = 6.0
    num_inference_steps = 30
    lora_weight = 0.55
    negative_prompt = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry"

    sample_size = [args.image_size, args.image_size]

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    config = OmegaConf.load(args.config_path)

    # Get Transformer3d
    transformer3d = Transformer3DModel.from_pretrained_2d(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        transformer_additional_kwargs=OmegaConf.to_container(
            config['transformer_additional_kwargs'])).to(weight_dtype)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict[
            "state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    if OmegaConf.to_container(config['vae_kwargs'])['enable_magvit']:
        Choosen_AutoencoderKL = AutoencoderKLMagvit
    else:
        Choosen_AutoencoderKL = AutoencoderKL
    vae = Choosen_AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype)

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict[
            "state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM": DDIMScheduler,
    }[args.sampler_name]
    scheduler = Choosen_Scheduler(
        **OmegaConf.to_container(config['noise_scheduler_kwargs']))

    pipeline = EasyAnimatePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        transformer=transformer3d,
        scheduler=scheduler,
        torch_dtype=weight_dtype)
    pipeline.to("cuda")
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    if args.lora_path is not None:
        pipeline = merge_lora(pipeline, args.lora_path, lora_weight)

    with open(args.prompt_info_path) as f:
        prompt_dicts = get_chunk(json.load(f), args.chunk_id, args.chunks_num)
        prompts = [d["prompt_en"] for d in prompt_dicts]
        prompt_batches = get_batch(prompts, args.batch_size)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    with torch.no_grad():
        for prompts in tqdm(prompt_batches):
            for i in range(args.video_num_per_prompt):
                samples = pipeline(
                    prompts,
                    video_length=video_length,
                    negative_prompt=negative_prompt,
                    height=sample_size[0],
                    width=sample_size[1],
                    generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                ).videos
                for prompt, sample in zip(prompts, samples):
                    video_path = os.path.join(args.save_path,
                                              f"{prompt}-{str(i)}.mp4")
                    save_videos_grid(sample.unsqueeze(0), video_path, fps=fps)


if __name__ == "__main__":
    main()
