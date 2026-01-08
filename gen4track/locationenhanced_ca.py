import torch
import torch.nn.functional as F
import math
from collections.abc import Iterable
import warnings
from utils import sam_utils

# Default cross-attention layers and heads used for guidance
DEFAULT_GUIDANCE_ATTN_KEYS = [("mid", 0, 0, 0), ("up", 1, 0, 0), ("up", 1, 1, 0), ("up", 1, 2, 0)]

def get_phrase_indices(tokenizer, prompt, phrases, verbose=False, words=None, include_eos=False, token_map=None, return_word_token_indices=False, add_suffix_if_not_found=False):
    """
    Find token indices corresponding to object phrases in the prompt.

    Args:
        tokenizer: CLIP tokenizer.
        prompt: Full text prompt.
        phrases: List of object phrases to locate.
        words: Optional specific words for single-token selection.
        include_eos: If True, include EOS token in positions.
        return_word_token_indices: If True, return indices of specific words.
        add_suffix_if_not_found: If phrase not in prompt, append it with separator.

    Returns:
        object_positions: List of token index lists for each phrase.
        Optionally: word_token_indices and/or modified prompt.
    """
    # Append missing phrases to prompt
    for obj in phrases:
        if obj not in prompt:
            prompt += "| " + obj

    if token_map is None:
        token_map = get_token_map(tokenizer, prompt=prompt, verbose=verbose, padding="do_not_pad")
    token_map_str = " ".join(token_map)

    object_positions = []
    word_token_indices = []

    for obj_ind, obj in enumerate(phrases):
        phrase_token_map = get_token_map(tokenizer, prompt=obj, verbose=False, padding="do_not_pad")
        phrase_token_map = phrase_token_map[1:-1]  # Remove BOS and EOS
        phrase_token_map_str = " ".join(phrase_token_map)

        if verbose:
            print("Full str:", token_map_str, "Substr:", phrase_token_map_str, "Phrase:", phrases)

        # Find starting index of phrase in full token map
        obj_first_index = len(token_map_str[:token_map_str.index(phrase_token_map_str) - 1].split(" "))
        obj_position = list(range(obj_first_index, obj_first_index + len(phrase_token_map)))

        if include_eos:
            obj_position.append(token_map.index(tokenizer.eos_token))
        object_positions.append(obj_position)

        if return_word_token_indices:
            if words is None:
                so_token_index = object_positions[-1][-1]  # Last token of phrase
                print(f"Picking the last token \"{token_map[so_token_index]}\" ({so_token_index}) as attention token")
            else:
                word = words[obj_ind]
                word_token_map = get_token_map(tokenizer, prompt=word, verbose=verbose, padding="do_not_pad")
                so_token_index = obj_first_index + phrase_token_map.index(word_token_map[-2])

            if verbose:
                print("so_token_index:", so_token_index)
            word_token_indices.append(so_token_index)

    if return_word_token_indices:
        if add_suffix_if_not_found:
            return object_positions, word_token_indices, prompt
        return object_positions, word_token_indices

    if add_suffix_if_not_found:
        return object_positions, prompt

    return object_positions


def add_ca_loss_per_attn_map_to_loss(loss, attn_map, bboxes, object_positions, use_ratio_based_loss=False, fg_top_p=0.2, bg_top_p=0.2, fg_weight=1.0, bg_weight=1.0, verbose=False):
    """
    Compute cross-attention loss for a single attention map across frames and objects.

    Args:
        loss: Accumulated loss tensor.
        attn_map: Attention map of shape [n_frames, n_heads, spatial_tokens, seq_len].
        bboxes: List of bounding boxes per frame (supports multiple boxes per object).
        object_positions: List of token indices corresponding to each object phrase.
        use_ratio_based_loss: If True, use deprecated ratio-based loss; else use max-based loss.
        fg_top_p, bg_top_p: Proportion of top pixels to consider in foreground/background for max-based loss.
        fg_weight, bg_weight: Weights for foreground and background terms in max-based loss.

    Returns:
        Updated loss tensor.
    """
    n, b, i, j = attn_map.shape  # n: frames in chunk, b: heads, i: spatial tokens, j: sequence length
    H = W = int(math.sqrt(i))     # Spatial resolution (e.g., 8x8 for latent space)

    for obj_idx in range(len(bboxes[0])):  # Iterate over objects (assumes same number per frame)
        for frame_id in range(len(bboxes)):  # Iterate over frames
            obj_loss = 0
            mask = torch.zeros(size=(H, W), device="cuda")
            obj_boxes = bboxes[frame_id]

            # Support both single box and multiple boxes per object
            if not isinstance(obj_boxes[0], Iterable):
                obj_boxes = [obj_boxes]

            # Create binary mask from bounding box(es)
            for obj_box in obj_boxes:
                box = obj_box.tolist()
                mask = sam_utils.proportion_to_mask(box, H, W)

            # Compute loss for each token position of the current object
            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[frame_id, :, :, obj_position].reshape(b, H, W)

                if use_ratio_based_loss:
                    warnings.warn("Using ratio-based loss, which is deprecated. Max-based loss is recommended.")
                    # Ratio-based: Encourage attention mass to stay within mask
                    activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(dim=-1)
                    obj_loss += torch.mean((1 - activation_value) ** 2)
                else:
                    # Max-based: Encourage high attention on top foreground pixels and low on background
                    ca_map_obj_flat = attn_map[frame_id, :, :, obj_position]
                    k_fg = (mask.sum() * fg_top_p).long().clamp_(min=1)
                    k_bg = ((1 - mask).sum() * bg_top_p).long().clamp_(min=1)
                    mask_1d = mask.view(1, -1)

                    obj_loss += (1 - (ca_map_obj_flat * mask_1d).topk(k=k_fg).values.mean(dim=1)).sum(dim=0) * fg_weight
                    obj_loss += ((ca_map_obj_flat * (1 - mask_1d)).topk(k=k_bg).values.mean(dim=1)).sum(dim=0) * bg_weight

        # Average loss over token positions for this object
        loss += obj_loss / len(object_positions[obj_idx])

    return loss


def compute_ca_lossv3(saved_attn, bboxes, object_positions, guidance_attn_keys, ref_ca_saved_attns=None, ref_ca_last_token_only=True, ref_ca_word_token_only=False, word_token_indices=None, index=None, ref_ca_loss_weight=1.0, verbose=False, **kwargs):
    """
    Compute overall cross-attention loss across specified UNet layers.

    Args:
        saved_attn: Dictionary of saved attention maps from forward pass.
        bboxes: Bounding boxes per frame.
        object_positions: Token indices for each object.
        guidance_attn_keys: List of attention layers/heads to use for guidance.
        index: Current timestep (for logging).

    Returns:
        Mean cross-attention loss.
    """
    loss = torch.tensor(0.0, device="cuda")
    object_number = len(bboxes[0])
    if object_number == 0:
        return loss

    for attn_key in guidance_attn_keys:
        attn_map_integrated = saved_attn[attn_key].cuda() if not saved_attn[attn_key].is_cuda else saved_attn[attn_key]
        attn_map = attn_map_integrated.squeeze(0)

        # Ensure 4D shape [frames, heads, spatial, seq]
        if attn_map.dim() == 3:
            attn_map = attn_map.unsqueeze(0)

        loss = add_ca_loss_per_attn_map_to_loss(loss, attn_map, bboxes, object_positions, verbose=verbose, **kwargs)

    num_attn = len(guidance_attn_keys)
    if num_attn > 0:
        loss = loss / (object_number * num_attn)

    return loss


def location_enhanced_latents(scheduler, unet, cond_embeddings, index, bboxes, object_positions, t, latents, loss, loss_scale=30, loss_threshold=0.2, max_iter=5, max_index_step=10, cross_attention_kwargs=None, ref_ca_saved_attns=None, guidance_attn_keys=DEFAULT_GUIDANCE_ATTN_KEYS, verbose=True, clear_cache=False, **kwargs):
    """
    Apply cross-attention guidance to refine latents by gradient descent on attention loss.

    Args:
        scheduler: Noise scheduler.
        unet: UNet model.
        cond_embeddings: Text conditioning embeddings.
        index: Current timestep index.
        bboxes: Object bounding boxes.
        object_positions: Token positions for objects.
        t: Current timestep tensor.
        latents: Initial noisy latents to refine.
        loss: Initial loss (updated in-place).
        loss_scale: Scaling factor for loss.
        loss_threshold: Stop refinement when de-scaled loss falls below this.
        max_iter: Maximum refinement iterations per timestep.
        max_index_step: Only apply guidance before this timestep.

    Returns:
        Refined latents and final loss.
    """
    iteration = 0

    if index < max_index_step:
        if isinstance(max_iter, list):
            max_iter = max_iter[index] if len(max_iter) > index else max_iter[-1]

        if verbose:
            print(f"time index {index}, loss: {loss.item()/loss_scale:.3f} (de-scaled with scale {loss_scale:.1f}), loss threshold: {loss_threshold:.3f}")

        while (loss.item() / loss_scale > loss_threshold and iteration < max_iter and index < max_index_step):
            saved_attn = {}
            full_cross_attention_kwargs = {
                'save_attn_to_dict': saved_attn,
                'save_keys': guidance_attn_keys,
            }

            latents.requires_grad_(True)
            latent_model_input = scheduler.scale_model_input(latents, t)

            # Forward pass to capture attention maps
            unet(latent_model_input, t, encoder_hidden_states=cond_embeddings,
                 return_cross_attention_probs=False, cross_attention_kwargs=full_cross_attention_kwargs)

            # Compute attention loss
            loss = compute_ca_lossv3(saved_attn=saved_attn, bboxes=bboxes, object_positions=object_positions,
                                     guidance_attn_keys=guidance_attn_keys, ref_ca_saved_attns=ref_ca_saved_attns,
                                     index=index, verbose=verbose, **kwargs) * loss_scale

            if torch.isnan(loss):
                print("**Loss is NaN**")

            # Compute gradient and update latents
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
            latents.requires_grad_(False)

            if hasattr(scheduler, 'sigmas'):
                latents = latents - grad_cond * (scheduler.sigmas[index] ** 2)
            elif hasattr(scheduler, 'alphas_cumprod'):
                warnings.warn("Using guidance scaled with alphas_cumprod")
                alpha_prod_t = scheduler.alphas_cumprod[t]
                scale = (1 - alpha_prod_t) ** 0.5
                latents = latents - scale * grad_cond
            else:
                warnings.warn("No scaling in guidance is performed")
                latents = latents - grad_cond

            iteration += 1

            if clear_cache:
                sam_utils.free_memory()

            if verbose:
                print(f"time index {index}, loss: {loss.item()/loss_scale:.3f}, loss threshold: {loss_threshold:.3f}, iteration: {iteration}")

    return latents, loss


def get_token_map(tokenizer, prompt, verbose=False, padding="do_not_pad"):
    """Tokenize prompt and return list of token strings (without padding)."""
    fg_prompt_tokens = tokenizer([prompt], padding=padding, max_length=77, return_tensors="np")
    input_ids = fg_prompt_tokens['input_ids'][0]

    token_map = []
    for ind, item in enumerate(input_ids.tolist()):
        token = tokenizer._convert_id_to_token(item)
        if verbose:
            print(f"{ind}, {token} ({item})")
        token_map.append(token)

    return token_map


