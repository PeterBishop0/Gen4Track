import torch
from PIL import ImageDraw
import numpy as np
import gc

# Device for torch operations (typically GPU)
torch_device = "cuda"


def draw_box(pil_img, bboxes, phrases):
    """Draw bounding boxes and corresponding phrases on a PIL image."""
    draw = ImageDraw.Draw(pil_img)

    for obj_bbox, phrase in zip(bboxes, phrases):
        x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
        # Scale coordinates to 512x512 image size and draw rectangle
        draw.rectangle([int(x_0 * 512), int(y_0 * 512), int(x_1 * 512), int(y_1 * 512)], outline='red', width=5)
        # Draw text label near the top-left corner of the box
        draw.text((int(x_0 * 512) + 5, int(y_0 * 512) + 5), phrase, font=None, fill=(255, 0, 0))
    
    return pil_img


def get_centered_box(box, horizontal_center_only=True, vertical_placement='centered', vertical_center=0.5, floor_padding=None):
    """Adjust bounding box to be centered (horizontally and optionally vertically) in normalized [0,1] coordinates."""
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    
    # Center horizontally
    x_min_new = 0.5 - w / 2
    x_max_new = 0.5 + w / 2
    
    if horizontal_center_only:
        return [x_min_new, y_min, x_max_new, y_max]
    
    h = y_max - y_min
    
    if vertical_placement == 'centered':
        assert floor_padding is None, "Set vertical_placement to 'floor_padding' to use floor padding"
        y_min_new = vertical_center - h / 2
        y_max_new = vertical_center + h / 2
    elif vertical_placement == 'floor_padding':
        # Place box near the bottom with specified padding from the image floor
        y_max_new = 1 - floor_padding
        y_min_new = y_max_new - h
    else:
        raise ValueError(f"Unknown vertical placement: {vertical_placement}")
    
    return [x_min_new, y_min_new, x_max_new, y_max_new]


def proportion_to_mask(obj_box, H, W, use_legacy=False, return_np=False):
    """Convert normalized bounding box to binary mask of size (H, W)."""
    width_scale = W / 512
    height_scale = H / 512
    
    x_min, y_min, x_max, y_max = obj_box
    # Scale normalized coordinates to target resolution
    x_min = int(x_min * width_scale)  
    y_min = int(y_min * height_scale)  
    x_max = int(x_max * width_scale)  
    y_max = int(y_max * height_scale)  
    
    # Create empty mask
    if return_np:
        mask = np.zeros((H, W))
    else:
        mask = torch.zeros(H, W).to(torch_device)
    # Set region inside box to 1
    mask[y_min:y_max, x_min:x_max] = 1.
    return mask


def scale_proportion(obj_box, H, W, use_legacy=False):
    """Scale normalized bounding box [0,1] to pixel coordinates in (H, W) image."""
    if use_legacy:
        # Legacy mode: direct integer scaling (may bias toward top-left)
        x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
    else:
        # Improved mode: round width and height separately to preserve box size
        x_min, y_min = round(obj_box[0] * W), round(obj_box[1] * H)
        box_w, box_h = round((obj_box[2] - obj_box[0]) * W), round((obj_box[3] - obj_box[1]) * H)
        x_max, y_max = x_min + box_w, y_min + box_h
        
        # Clamp to image boundaries
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, W), min(y_max, H)
        
    return x_min, y_min, x_max, y_max


def binary_mask_to_box(mask, enlarge_box_by_one=True, w_scale=1, h_scale=1):
    """Convert binary mask to bounding box in scaled pixel coordinates."""
    if isinstance(mask, torch.Tensor):
        mask_loc = torch.where(mask)
    else:
        mask_loc = np.where(mask)
    height, width = mask.shape
    
    if len(mask_loc[0]) == 0:  # Handle empty mask
        raise ValueError('The mask is empty')
    
    if enlarge_box_by_one:
        # Expand box by one pixel on all sides (clamped to image size)
        ymin, ymax = max(min(mask_loc[0]) - 1, 0), min(max(mask_loc[0]) + 1, height)
        xmin, xmax = max(min(mask_loc[1]) - 1, 0), min(max(mask_loc[1]) + 1, width)
    else:
        ymin, ymax = min(mask_loc[0]), max(mask_loc[0])
        xmin, xmax = min(mask_loc[1]), max(mask_loc[1])
    
    # Scale coordinates if needed
    box = [xmin * w_scale, ymin * h_scale, xmax * w_scale, ymax * h_scale]
    return box


def binary_mask_to_box_mask(mask, to_device=True):
    """Convert binary mask to a tight rectangular mask covering the original bounding box."""
    box = binary_mask_to_box(mask)
    x_min, y_min, x_max, y_max = box
    
    H, W = mask.shape
    new_mask = torch.zeros(H, W)
    if to_device:
        new_mask = new_mask.to(torch_device)
    # Fill rectangular region
    new_mask[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = 1.
    
    return new_mask


def binary_mask_to_center(mask, normalize=False):
    """Compute the center of mass of a binary mask."""
    h, w = mask.shape
    
    total = mask.sum()
    if total == 0:
        return 0.5, 0.5  # Default center if mask is empty
    
    if isinstance(mask, torch.Tensor):
        x_coord = ((mask.sum(dim=0) @ torch.arange(w, device=mask.device)) / total).item()
        y_coord = ((mask.sum(dim=1) @ torch.arange(h, device=mask.device)) / total).item()
    else:
        x_coord = (mask.sum(axis=0) @ np.arange(w)) / total
        y_coord = (mask.sum(axis=1) @ np.arange(h)) / total
    
    if normalize:
        x_coord, y_coord = x_coord / w, y_coord / h
    return x_coord, y_coord


def iou(mask, masks, eps=1e-6):
    """Compute Intersection over Union between one mask and multiple masks."""
    mask = mask[None].astype(bool)
    masks = masks.astype(bool)
    intersection = (mask & masks).sum(axis=(1, 2))
    union = (mask | masks).sum(axis=(1, 2))
    
    return intersection / (union + eps)


def free_memory():
    """Clear Python garbage and CUDA cache to free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def expand_overall_bboxes(overall_bboxes):
    """Flatten a list of lists of bounding boxes into a single flat list."""
    return sum(overall_bboxes, start=[])


def shift_tensor(tensor, x_offset, y_offset, base_w=8, base_h=8, offset_normalized=False, ignore_last_dim=False):
    """Shift a tensor spatially by given pixel offsets, preserving alignment with latent space."""
    if ignore_last_dim:
        tensor_h, tensor_w = tensor.shape[-3:-1]
    else:
        tensor_h, tensor_w = tensor.shape[-2:]
        
    if offset_normalized:
        # Convert normalized offset to aligned latent-space shift
        assert tensor_h % base_h == 0 and tensor_w % base_w == 0
        scale_from_base_h, scale_from_base_w = tensor_h // base_h, tensor_w // base_w
        x_offset = round(x_offset * base_w) * scale_from_base_w
        y_offset = round(y_offset * base_h) * scale_from_base_h
    
    # Create empty tensor for output
    new_tensor = torch.zeros_like(tensor)
    
    # Compute overlap region
    overlap_w = tensor_w - abs(x_offset)
    overlap_h = tensor_h - abs(y_offset)
    
    # Determine source and destination slice starts
    y_src_start = 0 if y_offset >= 0 else -y_offset
    y_dest_start = y_offset if y_offset >= 0 else 0
    
    x_src_start = 0 if x_offset >= 0 else -x_offset
    x_dest_start = x_offset if x_offset >= 0 else 0
    
    # Copy overlapping region
    if ignore_last_dim:
        new_tensor[..., y_dest_start:y_dest_start+overlap_h, x_dest_start:x_dest_start+overlap_w, :] = \
            tensor[..., y_src_start:y_src_start+overlap_h, x_src_start:x_src_start+overlap_w, :]
    else:
        new_tensor[..., y_dest_start:y_dest_start+overlap_h, x_dest_start:x_dest_start+overlap_w] = \
            tensor[..., y_src_start:y_src_start+overlap_h, x_src_start:x_src_start+overlap_w]

    return new_tensor
