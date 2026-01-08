import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
# from models import torch_device
from transformers import SamModel, SamProcessor
import vis
import sam_utils
import cv2
from scipy import ndimage
import os

# Global device for torch operations
torch_device = "cuda"


def load_sam():
    """Load the Segment Anything Model (SAM) ViT-Base and its processor."""
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(torch_device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    sam_model_dict = dict(
        sam_model=sam_model,
        sam_processor=sam_processor
    )

    return sam_model_dict


def sam(sam_model_dict, image, input_points=None, input_boxes=None, target_mask_shape=None, return_numpy=True):
    """
    Run Segment Anything Model (SAM) inference on one or more images.

    Args:
        sam_model_dict: Dictionary containing SAM model and processor.
        image: List of images, each [512, 512, 3] in range 0-255.
        input_points: Optional point prompts.
        input_boxes: Optional bounding box prompts (normalized xyxy format).
        target_mask_shape: Desired output mask shape (H, W).
        return_numpy: If True, return masks as numpy arrays; else torch tensors.

    Returns:
        masks: List of predicted masks.
        conf_scores: IoU prediction scores for the masks.
    """
    sam_model, sam_processor = sam_model_dict['sam_model'], sam_model_dict['sam_processor']
    
    # Convert tuple-based boxes to lists if necessary for compatibility
    if input_boxes and isinstance(input_boxes[0], tuple):
        input_boxes = [list(input_box) for input_box in input_boxes]
        
    if input_boxes and input_boxes[0] and isinstance(input_boxes[0][0], tuple):
        input_boxes = [[list(input_box) for input_box in input_boxes_item] for input_boxes_item in input_boxes]
    
    with torch.no_grad():
        with torch.autocast(torch_device):
            # Prepare inputs for SAM
            inputs = sam_processor(image, input_points=input_points, input_boxes=input_boxes, return_tensors="pt").to(torch_device)
            outputs = sam_model(**inputs)
        
        # Post-process masks to original image size
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu().float(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        conf_scores = outputs.iou_scores.cpu().numpy()[0, 0]
        del inputs, outputs
    
    # Optionally resize masks to target shape and convert type
    if return_numpy:
        masks = [F.interpolate(masks_item.type(torch.float), target_mask_shape, mode='bilinear').type(torch.bool).numpy() for masks_item in masks]
    else:
        masks = [F.interpolate(masks_item.type(torch.float), target_mask_shape, mode='bilinear').type(torch.bool) for masks_item in masks]

    return masks, conf_scores


def sam_point_input(sam_model_dict, image, input_points, **kwargs):
    """Run SAM with point prompts."""
    return sam(sam_model_dict, image, input_points=input_points, **kwargs)


def sam_box_input(sam_model_dict, image, input_boxes, **kwargs):
    """Run SAM with bounding box prompts."""
    return sam(sam_model_dict, image, input_boxes=input_boxes, **kwargs)


def get_iou_with_resize(mask, masks, masks_shape):
    """Compute IoU after resizing predicted masks to the reference mask shape."""
    masks = np.array([cv2.resize(mask.astype(np.uint8) * 255, masks_shape[::-1], cv2.INTER_LINEAR).astype(bool) for mask in masks])
    return sam_utils.iou(mask, masks)


def select_mask(masks, conf_scores, coarse_ious=None, rule="largest_over_conf", discourage_mask_below_confidence=0.85, discourage_mask_below_coarse_iou=0.2, verbose=True):
    """
    Select the best mask from multiple SAM predictions.

    Args:
        masks: Numpy boolean arrays of candidate masks.
        conf_scores: Confidence scores from SAM.
        coarse_ious: Optional coarse IoU with input bounding box mask.
        rule: Selection strategy (currently only "largest_over_conf").

    Returns:
        Selected mask and its confidence score.
    """
    mask_sizes = masks.sum(axis=(1, 2))
    
    if rule == "largest_over_conf":
        # Prefer largest mask, penalize low confidence or low coarse IoU
        max_mask_size = np.max(mask_sizes)
        if coarse_ious is not None:
            scores = mask_sizes - (conf_scores < discourage_mask_below_confidence) * max_mask_size - (coarse_ious < discourage_mask_below_coarse_iou) * max_mask_size
        else:
            scores = mask_sizes - (conf_scores < discourage_mask_below_confidence) * max_mask_size
        if verbose:
            print(f"mask_sizes: {mask_sizes}, scores: {scores}")
    else:
        raise ValueError(f"Unknown rule: {rule}")

    mask_id = np.argmax(scores)
    mask = masks[mask_id]
    selection_conf = conf_scores[mask_id]
    
    selection_coarse_iou = coarse_ious[mask_id] if coarse_ious is not None else None

    if verbose:
        print(f"Selected a mask with confidence: {selection_conf}, coarse_iou: {selection_coarse_iou}")

    # Optional verbose visualization of top 3 masks
    if verbose >= 2:
        plt.figure(figsize=(10, 8))
        for ind in range(3):
            plt.subplot(1, 3, ind + 1)
            plt.title(f"Mask {ind}, score {scores[ind]}, conf {conf_scores[ind]:.2f}, iou {coarse_ious[ind] if coarse_ious is not None else None:.2f}")
            plt.imshow(masks[ind])
        plt.tight_layout()
        plt.show()
        plt.savefig('plotsam.jpg')
        plt.close()

    return mask, selection_conf


def preprocess_mask(token_attn_np_smooth, mask_th, n_erode_dilate_mask=0):
    """Preprocess attention map to create a binary mask with optional morphological operations."""
    # Normalize attention map
    token_attn_np_smooth_normalized = token_attn_np_smooth - token_attn_np_smooth.min()
    token_attn_np_smooth_normalized /= token_attn_np_smooth_normalized.max()
    mask_thresholded = token_attn_np_smooth_normalized > mask_th
    
    # Apply erosion followed by dilation if specified
    if n_erode_dilate_mask:
        mask_thresholded = ndimage.binary_erosion(mask_thresholded, iterations=n_erode_dilate_mask)
        mask_thresholded = ndimage.binary_dilation(mask_thresholded, iterations=n_erode_dilate_mask)
    
    return mask_thresholded


def sam_refine_box(sam_input_image, box, *args, **kwargs):
    """Refine a single bounding box using SAM (convenience wrapper for one image and one box)."""
    box = box.tolist()  # Convert tensor to list if needed
    sam_input_images, boxes = [sam_input_image], [[box]]
    mask_selected_batched_list, conf_score_selected_batched_list = sam_refine_boxes(sam_input_images, boxes, *args, **kwargs)

    return mask_selected_batched_list[0][0], conf_score_selected_batched_list[0][0]


def sam_refine_boxes(sam_input_images, boxes, model_dict, height, width, H, W, discourage_mask_below_confidence, discourage_mask_below_coarse_iou, visualize=True):
    """
    Refine multiple bounding boxes per image using SAM and select best masks.

    Args:
        sam_input_images: List of input images.
        boxes: List of list of boxes (normalized) per image.
        model_dict: SAM model dictionary.
        height, width: Original image dimensions (512x512 typically).
        H, W: Target mask resolution.
        visualize: If True, save intermediate visualization.

    Returns:
        Selected masks and confidence scores per box.
    """
    # Run SAM with box prompts
    masks, conf_scores = sam_box_input(model_dict, image=sam_input_images, input_boxes=input_boxes, target_mask_shape=(H, W))
    
    mask_selected_batched_list, conf_score_selected_batched_list = [], []
    
    for boxes_item, masks_item in zip(boxes, masks):
        mask_selected_list, conf_score_selected_list = [], []
        for box, three_masks in zip(boxes_item, masks_item):
            # Create binary mask from input box for coarse IoU computation
            mask_binary = sam_utils.proportion_to_mask(box, H, W, return_np=True)
            if visualize:
                plt.title("Binary mask from input box (for iou)")
                plt.imshow(mask_binary)
                plt.savefig("mask.jpg")
            
            # Compute coarse IoU between input box mask and SAM predictions
            coarse_ious = get_iou_with_resize(mask_binary, three_masks, masks_shape=mask_binary.shape)

            # Select best mask among the three SAM outputs
            mask_selected, conf_score_selected = select_mask(three_masks, conf_scores, coarse_ious=coarse_ious, 
                                                            rule="largest_over_conf", 
                                                            discourage_mask_below_confidence=discourage_mask_below_confidence, 
                                                            discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
                                                            verbose=3)

            mask_selected_list.append(mask_selected)
            conf_score_selected_list.append(conf_score_selected)
        mask_selected_batched_list.append(mask_selected_list)
        conf_score_selected_batched_list.append(conf_score_selected_list)
    
    return mask_selected_batched_list, conf_score_selected_batched_list