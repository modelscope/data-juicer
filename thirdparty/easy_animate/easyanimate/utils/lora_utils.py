# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/bmaltais/kohya_ss

import hashlib
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import List, Optional, Type, Union

import safetensors.torch
import torch
import torch.utils.checkpoint
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from safetensors.torch import load_file
from transformers import T5EncoderModel


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim,
                                             self.lora_dim,
                                             kernel_size,
                                             stride,
                                             padding,
                                             bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim,
                                           out_dim, (1, 1), (1, 1),
                                           bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy(
            )  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x, *args, **kwargs):
        weight_dtype = x.dtype
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x.to(self.lora_down.weight.dtype))

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim),
                              device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout)
                                  )  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded.to(
            weight_dtype) + lx.to(weight_dtype) * self.multiplier * scale


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


class LoRANetwork(torch.nn.Module):
    TRANSFORMER_TARGET_REPLACE_MODULE = ["Transformer3DModel"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["T5LayerSelfAttention", "T5LayerFF"]
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder: Union[List[T5EncoderModel], T5EncoderModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        add_lora_in_attn_temporal: bool = False,
        varbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout

        print(
            f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}"
        )
        print(f"neuron dropout: p={self.dropout}")

        # create module instances
        def create_modules(
            is_unet: bool,
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:
            prefix = (self.LORA_PREFIX_TRANSFORMER
                      if is_unet else self.LORA_PREFIX_TEXT_ENCODER)
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "LoRACompatibleLinear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d" or child_module.__class__.__name__ == "LoRACompatibleConv"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (
                            1, 1)

                        if not add_lora_in_attn_temporal:
                            if "attn_temporal" in child_name:
                                continue

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None

                            if is_linear or is_conv2d_1x1:
                                dim = self.lora_dim
                                alpha = self.alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                            )
                            loras.append(lora)
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [
            text_encoder
        ]

        self.text_encoder_loras = []
        skipped_te = []
        for i, text_encoder in enumerate(text_encoders):
            text_encoder_loras, skipped = create_modules(
                False, text_encoder,
                LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped
        print(
            f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules."
        )

        self.unet_loras, skipped_un = create_modules(
            True, unet, LoRANetwork.TRANSFORMER_TARGET_REPLACE_MODULE)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def apply_to(self,
                 text_encoder,
                 unet,
                 apply_text_encoder=True,
                 apply_unet=True):
        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        info = self.load_state_dict(weights_sd, False)
        return info

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        return all_params

    def enable_gradient_checkpointing(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = precalculate_safetensors_hashes(
                state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    text_encoder: Union[T5EncoderModel, List[T5EncoderModel]],
    transformer,
    neuron_dropout: Optional[float] = None,
    add_lora_in_attn_temporal: bool = False,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    network = LoRANetwork(
        text_encoder,
        transformer,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        add_lora_in_attn_temporal=add_lora_in_attn_temporal,
        varbose=True,
    )
    return network


def merge_lora(pipeline,
               lora_path,
               multiplier,
               device='cpu',
               dtype=torch.float32,
               state_dict=None,
               transformer_only=False):
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if state_dict is None:
        state_dict = load_file(lora_path, device=device)
    else:
        state_dict = state_dict
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if "lora_te" in layer:
            if transformer_only:
                continue
            else:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER +
                                          "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_TRANSFORMER +
                                      "_")[-1].split("_")
            curr_layer = pipeline.transformer

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    print('Error loading layer')
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2),
                weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(
                weight_up, weight_down)

    return pipeline


# TODO: Refactor with merge_lora.
def unmerge_lora(pipeline,
                 lora_path,
                 multiplier=1,
                 device="cpu",
                 dtype=torch.float32):
    """Unmerge state_dict in LoRANetwork from the pipeline in diffusers."""
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    state_dict = load_file(lora_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if "lora_te" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER +
                                      "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.transformer

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    print('Error loading layer')
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        if len(weight_up.shape) == 4:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2),
                weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(
                weight_up, weight_down)

    return pipeline
