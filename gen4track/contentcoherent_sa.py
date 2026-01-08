from __future__ import annotations
import math
import time
from typing import Type, Dict, Any, Tuple, Callable

import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F

from . import merge
from .utils import isinstance_str, init_generator, join_frame, split_frame, func_warper, join_warper, split_warper

from dataclasses import dataclass
from diffusers import StableDiffusionControlNetPipeline
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
import einops

T = torch.Tensor


@dataclass(frozen=True)
class StyleAlignedArgs:
    """Configuration for StyleAligned attention and normalization sharing."""
    share_group_norm: bool = True
    share_layer_norm: bool = True
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1.
    shared_score_shift: float = 0.
    only_self_level: float = 0.


def compute_merge(module: torch.nn.Module, x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    """Compute token merge and unmerge operations for Token Merging in video/multi-frame setting."""
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    generator = module.generator

    # Frame and token counts
    fsize = x.shape[0] // args["batch_size"]
    tsize = x.shape[1]

    # Apply merging only in higher resolution layers
    if downsample <= args["max_downsample"]:
        # Initialize random generator if needed
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # Local (per-frame) token merging
        local_tokens = join_frame(x, fsize)
        m_ls = [join_warper(fsize)]
        u_ls = [split_warper(fsize)]
        unm = 0
        curF = fsize

        # Recursively merge across frames until target stride is reached
        while curF > 1:
            m, u, ret_dict = merge.bipartite_soft_matching_randframe(
                local_tokens, curF, args["local_merge_ratio"], unm, generator, args["target_stride"], args["align_batch"])
            unm += ret_dict["unm_num"]
            m_ls.append(m)
            u_ls.append(u)
            local_tokens = m(local_tokens)
            curF = (local_tokens.shape[1] - unm) // tsize

        merged_tokens = local_tokens

        # Optional global token merging with stored global tokens
        if args["merge_global"]:
            if hasattr(module, "global_tokens") and module.global_tokens is not None:
                if torch.rand(1, generator=generator, device=generator.device) > args["global_rand"]:
                    src_len = local_tokens.shape[1]
                    tokens = torch.cat([local_tokens, module.global_tokens.to(local_tokens)], dim=1)
                    local_chunk = 0
                else:
                    src_len = module.global_tokens.shape[1]
                    tokens = torch.cat([module.global_tokens.to(local_tokens), local_tokens], dim=1)
                    local_chunk = 1

                m, u, _ = merge.bipartite_soft_matching_2s(
                    tokens, src_len, args["global_merge_ratio"], args["align_batch"], unmerge_chunk=local_chunk)
                merged_tokens = m(tokens)
                m_ls.append(m)
                u_ls.append(u)
                module.global_tokens = u(merged_tokens).detach().clone().cpu()
            else:
                module.global_tokens = local_tokens.detach().clone().cpu()

        m = func_warper(m_ls)
        u = func_warper(u_ls[::-1])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)
        merged_tokens = x

    return m, u, merged_tokens


def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """Patch a transformer block to apply Token Merging (ToMe) during forward pass."""
    class ToMeBlock(block_class):
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(self, x, self._tome_info)

            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x

    return ToMeBlock


def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """Patch a diffusers-style transformer block to apply Token Merging."""
    class ToMeBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # Normalize hidden states based on ADA norm variants
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # Apply token merging to normalized states
            m_a, u_a, merged_tokens = compute_merge(self, norm_hidden_states, self._tome_info)
            norm_hidden_states = merged_tokens

            # Self-attention
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **(cross_attention_kwargs or {}),
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # Unmerge and add residual
            attn_output = u_a(attn_output)
            hidden_states = attn_output + hidden_states

            # Cross-attention (if present)
            if self.attn2 is not None:
                norm_hidden_states = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **(cross_attention_kwargs or {}),
                )
                hidden_states = attn_output + hidden_states

            # Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states
            return hidden_states

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """Register pre-hook to capture input image size for ToMe."""
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_tome_module(module: torch.nn.Module):
    """Register pre-hook to initialize shared random generator for consistent merging."""
    def hook(module, args):
        if not hasattr(module, "generator"):
            module.generator = init_generator(args[0].device)
        elif module.generator.device != args[0].device:
            module.generator = init_generator(args[0].device, fallback=module.generator)
        return None

    module._tome_info["hooks"].append(module.register_forward_pre_hook(hook))


def update_patch(model: torch.nn.Module, **kwargs):
    """Update ToMe arguments across all patched modules in the model (including ControlNet)."""
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for k, v in kwargs.items():
                    setattr(module, k, v)
    return model


def collect_from_patch(model: torch.nn.Module, attr="tome"):
    """Collect specific attributes from all patched modules."""
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    ret_dict = dict()
    for model in model_ls:
        for name, module in model.named_modules():
            if hasattr(module, attr):
                res = getattr(module, attr)
                ret_dict[name] = res
    return ret_dict


def expand_first(feat: T, scale=1.) -> T:
    """Expand reference style features from first and middle frames to entire batch."""
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.) -> T:
    """Concatenate expanded style features to original features along specified dimension."""
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    """Compute mean and standard deviation along spatial dimensions."""
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    """Apply Adaptive Instance Normalization using expanded reference statistics."""
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


class DefaultAttentionProcessor(nn.Module):
    """Standard attention processor using Diffusers' AttnProcessor2_0."""

    def __init__(self):
        super().__init__()
        self.processor = attention_processor.AttnProcessor2_0()

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)


class SharedAttentionProcessor(DefaultAttentionProcessor):
    """Custom attention processor for StyleAligned: shares attention across reference and target."""

    def shifted_scaled_dot_product_attention(self, attn: attention_processor.Attention, query: T, key: T, value: T) -> T:
        logits = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale
        logits[:, :, :, query.shape[2]:] += self.shared_score_shift
        probs = logits.softmax(-1)
        return torch.einsum('bhqk,bhkd->bhqd', probs, value)

    def shared_call(
            self,
            attn: attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
    ):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply AdaIN to queries, keys, values if enabled
        if self.adain_queries:
            query = adain(query)
        if self.adain_keys:
            key = adain(key)
        if self.adain_values:
            value = adain(value)

        # Share attention between reference and target
        if self.share_attention:
            key = concat_first(key, -2, scale=self.shared_score_scale)
            value = concat_first(value, -2)
            if self.shared_score_shift != 0:
                hidden_states = self.shifted_scaled_dot_product_attention(attn, query, key, value)
            else:
                hidden_states = nnf.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        else:
            hidden_states = nnf.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        if self.full_attention_share:
            b, n, d = hidden_states.shape
            hidden_states = einops.rearrange(hidden_states, '(k b) n d -> k (b n) d', k=2)
            hidden_states = super().__call__(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                             attention_mask=attention_mask, **kwargs)
            hidden_states = einops.rearrange(hidden_states, 'k (b n) d -> (k b) n d', n=n)
        else:
            hidden_states = self.shared_call(attn, hidden_states, hidden_states, attention_mask, **kwargs)
        return hidden_states

    def __init__(self, style_aligned_args: StyleAlignedArgs):
        super().__init__()
        self.share_attention = style_aligned_args.share_attention
        self.adain_queries = style_aligned_args.adain_queries
        self.adain_keys = style_aligned_args.adain_keys
        self.adain_values = style_aligned_args.adain_values
        self.full_attention_share = style_aligned_args.full_attention_share
        self.shared_score_scale = style_aligned_args.shared_score_scale
        self.shared_score_shift = style_aligned_args.shared_score_shift


def _get_switch_vec(total_num_layers, level):
    """Generate boolean vector indicating which self-attention layers should use shared attention."""
    if level == 0:
        return torch.zeros(total_num_layers, dtype=torch.bool)
    if level == 1:
        return torch.ones(total_num_layers, dtype=torch.bool)
    to_flip = level > .5
    if to_flip:
        level = 1 - level
    num_switch = int(level * total_num_layers)
    vec = torch.arange(total_num_layers) % (total_num_layers // num_switch) == 0
    if to_flip:
        vec = ~vec
    return vec


def init_attention_processors(pipeline: StableDiffusionControlNetPipeline, style_aligned_args: StyleAlignedArgs | None = None):
    """Set custom attention processors (default or shared) in UNet based on StyleAligned configuration."""
    attn_procs = {}
    unet = pipeline.unet
    num_self_layers = len([name for name in unet.attn_processors.keys() if 'attn1' in name])
    only_self_vec = _get_switch_vec(num_self_layers, 1 if style_aligned_args is None else style_aligned_args.only_self_level)

    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attention = 'attn1' in name
        if is_self_attention:
            if style_aligned_args is None or only_self_vec[i // 2]:
                attn_procs[name] = DefaultAttentionProcessor()
            else:
                attn_procs[name] = SharedAttentionProcessor(style_aligned_args)
        else:
            attn_procs[name] = DefaultAttentionProcessor()

    unet.set_attn_processor(attn_procs)


def register_shared_norm(pipeline: StableDiffusionControlNetPipeline,
                         share_group_norm: bool = True,
                         share_layer_norm: bool = True):
    """Patch GroupNorm and LayerNorm to share statistics from reference frames."""
    def register_norm_forward(norm_layer: nn.GroupNorm | nn.LayerNorm) -> nn.GroupNorm | nn.LayerNorm:
        if not hasattr(norm_layer, 'orig_forward'):
            setattr(norm_layer, 'orig_forward', norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T:
            n = hidden_states.shape[-2]
            hidden_states = concat_first(hidden_states, dim=-2)
            hidden_states = orig_forward(hidden_states)
            return hidden_states[..., :n, :]

        norm_layer.forward = forward_
        return norm_layer

    def get_norm_layers(module_, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]):
        if isinstance(module_, nn.LayerNorm) and share_layer_norm:
            norm_layers_['layer'].append(module_)
        if isinstance(module_, nn.GroupNorm) and share_group_norm:
            norm_layers_['group'].append(module_)
        else:
            for layer in module_.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {'group': [], 'layer': []}
    get_norm_layers(pipeline.unet, norm_layers)
    return [register_norm_forward(layer) for layer in norm_layers['group']] + [register_norm_forward(layer) for layer in norm_layers['layer']]


class ContentCoherentPatchController:
    """
    Unified controller for applying:
    - Token Merging (ToMe) for efficiency in video generation
    - StyleAligned (shared norm + shared attention) for style consistency
    """

    def __init__(self, pipeline: StableDiffusionControlNetPipeline):
        self.pipeline = pipeline
        self.norm_layers: list[nn.Module] = []
        self.stylealign_enabled = False
        self.token_merge_enabled = False

    def register_stylealign(self, style_aligned_args: StyleAlignedArgs):
        """Enable StyleAligned by patching norms and attention processors."""
        if self.stylealign_enabled:
            return
        self.norm_layers = register_shared_norm(
            self.pipeline,
            style_aligned_args.share_group_norm,
            style_aligned_args.share_layer_norm,
        )
        init_attention_processors(self.pipeline, style_aligned_args)
        self.stylealign_enabled = True

    def remove_stylealign(self):
        """Restore original norm forward and default attention processors."""
        if not self.stylealign_enabled:
            return
        for layer in self.norm_layers:
            layer.forward = layer.orig_forward
        self.norm_layers.clear()
        init_attention_processors(self.pipeline, None)
        self.stylealign_enabled = False

    def register_token_merge(
        self,
        local_merge_ratio: float = 0.9,
        merge_global: bool = False,
        global_merge_ratio: float = 0.8,
        max_downsample: int = 2,
        seed: int = 123,
        batch_size: int = 2,
        include_control: bool = False,
        align_batch: bool = False,
        target_stride: int = 4,
        global_rand: float = 0.5,
    ):
        """Enable Token Merging (ToMe) by patching transformer blocks."""
        if self.token_merge_enabled:
            return

        self.remove_token_merge()  # Ensure clean state

        model = self.pipeline
        is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

        diffusion_model = model.unet if hasattr(model, "unet") else model.model.diffusion_model
        diffusion_models = [diffusion_model]
        if isinstance_str(model, "StableDiffusionControlNetPipeline") and include_control:
            diffusion_models.append(model.controlnet)

        for diffusion_model in diffusion_models:
            diffusion_model._tome_info = {
                "size": None,
                "hooks": [],
                "args": {
                    "max_downsample": max_downsample,
                    "generator": None,
                    "seed": seed,
                    "batch_size": batch_size,
                    "align_batch": align_batch,
                    "merge_global": merge_global,
                    "global_merge_ratio": global_merge_ratio,
                    "local_merge_ratio": local_merge_ratio,
                    "global_rand": global_rand,
                    "target_stride": target_stride,
                },
            }

            hook_tome_model(diffusion_model)

            for name, module in diffusion_model.named_modules():
                if isinstance_str(module, "BasicTransformerBlock"):
                    make_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
                    module.__class__ = make_block_fn(module.__class__)
                    module._tome_info = diffusion_model._tome_info
                    hook_tome_module(module)

                    if not hasattr(module, "disable_self_attn") and not is_diffusers:
                        module.disable_self_attn = False
                    if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                        module.use_ada_layer_norm = False
                        module.use_ada_layer_norm_zero = False

        self.token_merge_enabled = True

    def remove_token_merge(self):
        """Restore original transformer blocks and remove hooks."""
        model = self.pipeline.unet if hasattr(self.pipeline, "unet") else self.pipeline
        model_ls = [model]
        if hasattr(self.pipeline, "controlnet"):
            model_ls.append(self.pipeline.controlnet)

        for model in model_ls:
            for _, module in model.named_modules():
                if hasattr(module, "_tome_info"):
                    for hook in module._tome_info["hooks"]:
                        hook.remove()
                    module._tome_info["hooks"].clear()

                if module.__class__.__name__ == "ToMeBlock":
                    module.__class__ = module._parent

        self.token_merge_enabled = False

    def register_all(self, style_aligned_args: StyleAlignedArgs, **token_merge_kwargs):
        """Enable both StyleAligned and Token Merging."""
        self.register_stylealign(style_aligned_args)
        self.register_token_merge(**token_merge_kwargs)

    def remove_all(self):
        """Disable both patches."""
        self.remove_stylealign()
        self.remove_token_merge()
