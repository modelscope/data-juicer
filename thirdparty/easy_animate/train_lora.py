"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import logging
import math
import os
import re
import shutil
import sys

import accelerate
# import easyanimate pakage
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from transformers.utils import ContextManagers

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from easyanimate.data.bucket_sampler import (ASPECT_RATIO_512,
                                             ASPECT_RATIO_RANDOM_CROP_512,
                                             ASPECT_RATIO_RANDOM_CROP_PROB,
                                             AspectRatioBatchSampler,
                                             get_closest_ratio)
from easyanimate.data.dataset_video import VideoDataset
from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer3d import Transformer3DModel
from easyanimate.pipeline.pipeline_easyanimate import EasyAnimatePipeline
from easyanimate.utils.IDDIM import IDDPM
from easyanimate.utils.lora_utils import (create_network, merge_lora,
                                          unmerge_lora)
from easyanimate.utils.utils import save_videos_grid

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def auto_scale_lr(effective_bs, lr, rule='linear', base_batch_size=256):
    assert rule in ['linear', 'sqrt']
    # scale by world size
    if rule == 'sqrt':
        scale_ratio = math.sqrt(effective_bs / base_batch_size)
    elif rule == 'linear':
        scale_ratio = effective_bs / base_batch_size
    lr *= scale_ratio
    logger.info(
        f'Automatically adapt lr to {lr:.7f} (using {rule} scaling rule).')
    return lr


def log_validation(vae, text_encoder, tokenizer, transformer3d, network,
                   config, args, accelerator, weight_dtype, global_step):
    # try:
    logger.info("Running validation... ")

    transformer3d_val = Transformer3DModel.from_pretrained_2d(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        transformer_additional_kwargs=OmegaConf.to_container(
            config['transformer_additional_kwargs'])).to(weight_dtype)
    transformer3d_val.load_state_dict(
        accelerator.unwrap_model(transformer3d).state_dict())

    pipeline = EasyAnimatePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae).to(weight_dtype),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        transformer=transformer3d_val,
        torch_dtype=weight_dtype)
    pipeline = pipeline.to(accelerator.device)
    pipeline = merge_lora(
        pipeline,
        None,
        1,
        accelerator.device,
        state_dict=accelerator.unwrap_model(network).state_dict(),
        transformer_only=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(
            args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.no_grad():
            sample = pipeline(args.validation_prompts[i],
                              video_length=args.sample_n_frames,
                              negative_prompt="bad detailed",
                              height=args.sample_size,
                              width=args.sample_size,
                              generator=generator).videos
        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
        save_videos_grid(
            sample,
            os.path.join(args.output_dir,
                         f"sample/sample-{global_step}-{i}.gif"))

    del pipeline
    del transformer3d_val
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=
        "Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=
        "Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=("A csv containing the training data. "),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=
        ("For debugging purposes or quicker training, truncate the number of training examples to this "
         "value if set."),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=
        ("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."
         ),
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        nargs="+",
        help=
        ("A set of images evaluated every `--validation_epochs` and logged to `--report_to`."
         ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=
        "The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--vae_mini_batch",
                        type=int,
                        default=32,
                        help="mini batch size for vae.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=
        ("Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
         " remote repository specified with --pretrained_model_name_or_path."),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=
        ("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
         ),
    )
    parser.add_argument("--adam_beta1",
                        type=float,
                        default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2",
                        type=float,
                        default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay",
                        type=float,
                        default=1e-2,
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon",
                        type=float,
                        default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--push_to_hub",
                        action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token",
                        type=str,
                        default=None,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help=
        "The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help=
        "The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=
        ("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
         ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=
        ('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
         ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
         ),
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention",
                        action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=
        ("The `project_name` argument passed to Accelerator.init_trackers for"
         " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
         ),
    )
    parser.add_argument(
        "--tracker_experiment_name",
        type=str,
        default="experiment1",
        help=
        ("The name for wandb init"),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help=
        "Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument("--snr_loss",
                        action="store_true",
                        help="Whether or not to use snr_loss.")
    parser.add_argument(
        "--random_ratio_crop",
        action="store_true",
        help="Whether enable random ratio crop sample in datasets.")
    parser.add_argument(
        "--random_frame_crop",
        action="store_true",
        help="Whether enable random frame crop sample in datasets.")
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=256,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--sample_n_frames",
        type=int,
        default=4,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=("The config of the model in training."),
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

    parser.add_argument('--tokenizer_max_length',
                        type=int,
                        default=120,
                        help='Max length of tokenizer')

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    
    if args.report_to == "wandb":
        wandb.init(project=args.tracker_project_name, name=args.tracker_experiment_name)

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=
            ("Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
             " use `--variant=non_ema` instead."),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)

    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2,
                                       gradient_accumulation_steps=1)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin)
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    train_diffusion = IDDPM(str(args.train_sampling_steps),
                            learn_sigma=True,
                            pred_sigma=True,
                            snr=args.snr_loss)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder="tokenizer",
                                            revision=args.revision)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState(
        ).deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    if OmegaConf.to_container(config['vae_kwargs'])['enable_magvit']:
        Choosen_AutoencoderKL = AutoencoderKLMagvit
    else:
        Choosen_AutoencoderKL = AutoencoderKL

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype)
        vae = Choosen_AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant)

    transformer3d = Transformer3DModel.from_pretrained_2d(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        transformer_additional_kwargs=OmegaConf.to_container(
            config['transformer_additional_kwargs']))

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    # Lora will work with this...
    network = create_network(
        1.0,
        args.rank,
        args.network_alpha,
        text_encoder,
        transformer3d,
        neuron_dropout=None,
        add_lora_in_attn_temporal=True,
    )
    network.apply_to(text_encoder, transformer3d, args.train_text_encoder,
                     True)

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
        assert len(u) == 0

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
        assert len(u) == 0

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer3d.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                models[0].save_pretrained(
                    os.path.join(output_dir, "transformer"))
                weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = Transformer3DModel.from_pretrained_2d(
                    input_dir,
                    subfolder="transformer",
                    transformer_additional_kwargs=OmegaConf.to_container(
                        config['transformer_additional_kwargs']))
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = auto_scale_lr(
            args.gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes, args.learning_rate, 'sqrt')

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    logging.info("Add network parameters")
    trainable_params = list(
        filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(
        args.learning_rate / 2, args.learning_rate, args.learning_rate)

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = VideoDataset(args.train_data_meta,
                                 args.train_data_dir,
                                 sample_size=args.sample_size,
                                 sample_stride=args.sample_stride,
                                 sample_n_frames=args.sample_n_frames,
                                 enable_inpaint=False)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, optimizer, train_dataloader, lr_scheduler)

    # Move text_encode and vae to gpu and cast to weight_dtype
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("validation_images")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_compute_budget = args.num_train_epochs * len(
        train_dataset) * args.sample_n_frames * args.sample_size**2

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total compute budget = {total_compute_budget}")

    assert total_compute_budget <= 524288000000, f" total_compute_budget = {total_compute_budget} exceeds the upper limit 524288000000, please readjust epochs, dataset size or sample size."

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    log_validation(vae, text_encoder, tokenizer, transformer3d, network,
                   config, args, accelerator, weight_dtype, global_step)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(
                ), batch['text']
                pixel_values = rearrange(pixel_values,
                                         "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"),
                            exist_ok=True)
                for idx, (pixel_value,
                          text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(
                        pixel_value,
                        f"{args.output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'}.gif",
                        rescale=True)

            with accelerator.accumulate(network):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)

                if args.random_frame_crop:
                    select_frames = [
                        _tmp
                        for _tmp in list(range(4, args.sample_n_frames + 4, 4))
                    ]
                    select_frames_prob = [_tmp**2 for _tmp in select_frames]
                    select_frames_prob = np.array(select_frames_prob) / sum(
                        select_frames_prob)
                    select_frames_prob = np.array(select_frames) / sum(
                        select_frames)

                    temp_n_frames = np.random.choice(select_frames,
                                                     p=select_frames_prob)
                    pixel_values = pixel_values[:, :temp_n_frames, :, :]

                video_length = pixel_values.shape[1]
                with torch.no_grad():
                    # This way will be slow when batch grows up
                    # pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    # latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist
                    # latents = latents.sample()
                    # latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

                    # Convert images to latent space
                    if vae.quant_conv.weight.ndim == 5:
                        # This way is quicker when batch grows up
                        pixel_values = rearrange(pixel_values,
                                                 "b f c h w -> b c f h w")
                        mini_batch = 21
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[2], mini_batch):
                            pixel_values_bs = pixel_values[:, :, i:i +
                                                           mini_batch, :, :]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                            # if i == pixel_values.shape[2] - 1:
                            #     break
                            # with torch.no_grad():
                            #     pixel_values_bs = pixel_values[:, :, i: i + mini_batch + 1, :, :].to(dtype=weight_dtype)
                            #     pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            #     pixel_values_bs = pixel_values_bs.sample()
                            #     new_pixel_values.append(pixel_values_bs if i == 0 else pixel_values_bs[:, :, 1:, :, :])
                        latents = torch.cat(new_pixel_values, dim=2)
                    else:
                        # This way is quicker when batch grows up
                        pixel_values = rearrange(pixel_values,
                                                 "b f c h w -> (b f) c h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i:i + bs]
                            pixel_values_bs = vae.encode(
                                pixel_values_bs.to(
                                    dtype=weight_dtype)).latent_dist
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        latents = torch.cat(new_pixel_values, dim=0)
                        latents = rearrange(latents,
                                            "(b f) c h w -> b c f h w",
                                            f=video_length)

                    latents = latents * 0.18215

                prompt_ids = tokenizer(batch['text'],
                                       max_length=args.tokenizer_max_length,
                                       padding="max_length",
                                       add_special_tokens=True,
                                       truncation=True,
                                       return_tensors="pt")
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids.to(latents.device),
                    attention_mask=prompt_ids.attention_mask.to(
                        latents.device),
                    return_dict=False)[0]

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0,
                                          args.train_sampling_steps, (bsz, ),
                                          device=latents.device)
                timesteps = timesteps.long()

                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
                loss_term = train_diffusion.training_losses(
                    transformer3d,
                    latents,
                    timesteps,
                    model_kwargs=dict(
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=prompt_ids.attention_mask.to(
                            latents.device),
                        added_cond_kwargs=added_cond_kwargs,
                        inpaint_latents=None,
                        return_dict=False))
                loss = loss_term['loss'].mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item(
                ) / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params,
                                                args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints
                                if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints,
                                key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints
                                   ) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints
                                ) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[
                                    0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        safetensor_save_path = os.path.join(
                            args.output_dir,
                            f"checkpoint-{global_step}.safetensors")
                        accelerator_save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        save_model(safetensor_save_path,
                                   accelerator.unwrap_model(network))
                        #if args.save_state:
                        #accelerator.save_state(accelerator_save_path)
                        #logger.info(f"Saved state to {accelerator_save_path}")

                if accelerator.is_main_process:
                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            transformer3d,
                            network,
                            config,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    transformer3d,
                    network,
                    config,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        safetensor_save_path = os.path.join(
            args.output_dir, f"checkpoint-{global_step}.safetensors")
        accelerator_save_path = os.path.join(args.output_dir,
                                             f"checkpoint-{global_step}")
        save_model(safetensor_save_path, accelerator.unwrap_model(network))
        #if args.save_state:
        #accelerator.save_state(accelerator_save_path)
        #logger.info(f"Saved state to {accelerator_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
