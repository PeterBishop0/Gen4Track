import contextlib
import random
import numpy as np
import os
from glob import glob
from PIL import Image, ImageSequence
import imageio
import torch
from torchvision.io import read_video, write_video
import torchvision.transforms as T
import gc
import cv2
from diffusers import DDIMScheduler, StableDiffusionControlNetPipeline, StableDiffusionPipeline, StableDiffusionDepth2ImgPipeline, ControlNetModel
from .controlnet_utils import CONTROLNET_DICT, control_preprocess
from einops import rearrange

# Supported frame image extensions
FRAME_EXT = [".jpg", ".png"]


def init_model(device="cuda", sd_version="1.5", model_key=None, control_type="none", weight_dtype="fp16"):
    """Initialize Stable Diffusion pipeline with optional ControlNet or depth variant."""
    use_depth = False
    if model_key is None:
        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        elif sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
            use_depth = True
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        print(f'[INFO] loading stable diffusion from: {model_key}')
    else:
        print(f'[INFO] loading custom model from: {model_key}')

    # Load scheduler
    scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

    # Set precision
    if weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    # Load pipeline based on control type or depth variant
    if control_type not in ["none", "pnp"]:
        controlnet_key = CONTROLNET_DICT[control_type]
        print(f'[INFO] loading controlnet from: {controlnet_key}')
        controlnet = ControlNetModel.from_pretrained(controlnet_key, torch_dtype=weight_dtype)
        print(f'[INFO] loaded controlnet!')
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_key, controlnet=controlnet, torch_dtype=weight_dtype, local_files_only=False
        )
    elif use_depth:
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(model_key, torch_dtype=weight_dtype)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=weight_dtype)

    return pipe.to(device), scheduler, model_key


def seed_everything(seed):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_image(image_path):
    """Load a single image as a torch tensor in [0,1] range with batch dimension."""
    image = Image.open(image_path).convert('RGB')
    image = T.ToTensor()(image)
    return image.unsqueeze(0)


def process_frames(frames, h, w):
    """Resize and center-crop video frames to target resolution while preserving aspect ratio."""
    fh, fw = frames.shape[-2:]
    h = int(np.floor(h / 64.0)) * 64
    w = int(np.floor(w / 64.0)) * 64

    # Compute resize size that preserves aspect ratio
    nw = int(fw / fh * h)
    if nw >= w:
        size = (h, nw)
    else:
        size = (int(fh / fw * w), w)

    if len(frames.shape) == 3:
        frames = [frames]

    print(f"[INFO] frame size {(fh, fw)} resize to {size} and centercrop to {(h, w)}")

    frame_ls = []
    for frame in frames:
        resized_frame = T.Resize(size, antialias=True)(frame)
        cropped_frame = T.CenterCrop([h, w])(resized_frame)
        frame_ls.append(cropped_frame)

    return torch.stack(frame_ls), size


def process_bbox(bbox_path, original_size, resize_size, crop_size, frames_number, xyxy=False):
    """Adjust bounding boxes according to resize and center-crop operations."""
    total_bbox_ls = []
    id = 0
    with open(bbox_path, 'r') as f:
        for line in f:
            id += 1
            if id > frames_number:
                break
            bbox_str = line.replace(' ', ',').replace('\t', ',')
            bbox_list = bbox_str.strip('\n').split(',')
            bbox_list = list(map(int, bbox_list))
            bbox = np.array(bbox_list)
            if xyxy:
                total_bbox_ls.append(bbox)
            else:  # Convert xywh to xyxy
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                total_bbox_ls.append(bbox)
            
    if len(total_bbox_ls) != frames_number:
        print("bbox和帧数不相等")         
    
    adjusted_bbox_ls = []
    # Scaling factors after resize
    width_scale = resize_size[1] / original_size[1]  
    height_scale = resize_size[0] / original_size[0] 

    for bbox in total_bbox_ls:
        bbox = np.array(bbox, dtype=np.float32)  

        # Scale coordinates
        new_x1 = int(bbox[0] * width_scale)  
        new_y1 = int(bbox[1] * height_scale)  
        new_x2 = int(bbox[2] * width_scale)  
        new_y2 = int(bbox[3] * height_scale)  
        
        # Crop offsets (top-left corner of center crop in resized image)
        crop_x1 = (resize_size[1] - crop_size[1]) // 2 
        crop_y1 = (resize_size[0] - crop_size[0]) // 2 
        
        # Adjust for center crop
        new_x1 = max(new_x1 - crop_x1, 0)  
        new_y1 = max(new_y1 - crop_y1, 0)  
        new_x2 = min(new_x2 - crop_x1, crop_size[1])  
        new_y2 = min(new_y2 - crop_y1, crop_size[0]) 
        
        # Ensure valid box dimensions
        new_width = max(new_x2 - new_x1, 0)  
        new_height = max(new_y2 - new_y1, 0)  
        
        new_bbox = [new_x1, new_y1, new_x1 + new_width, new_y1 + new_height] 
        adjusted_bbox_ls.append(new_bbox)

    return torch.IntTensor(adjusted_bbox_ls)


def glob_frame_paths(video_path):
    """Collect and sort image frame paths from a directory."""
    frame_paths = []
    for ext in FRAME_EXT:
        frame_paths += glob(os.path.join(video_path, f"*{ext}"))
    frame_paths = sorted(frame_paths)
    return frame_paths


def load_video(video_path, bbox_path, h, w, frame_ids=None, device="cuda"):
    """Load video frames (from video file, GIF, or image sequence) and optional bounding boxes."""
    if ".mp4" in video_path:
        frames, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
        frames = frames / 255.0
    elif ".gif" in video_path:
        frames = Image.open(video_path)
        frame_ls = []
        for frame in ImageSequence.Iterator(frames):
            frame_ls += [T.ToTensor()(frame.convert("RGB"))]
        frames = torch.stack(frame_ls)
    else:
        frame_paths = glob_frame_paths(video_path)
        frame_ls = []
        for frame_path in frame_paths:
            frame = load_image(frame_path)
            frame_ls.append(frame)
        frames = torch.cat(frame_ls)

    if frame_ids is not None:
        frames = frames[frame_ids]

    print(f"[INFO] loaded video with {len(frames)} frames from: {video_path}")
    original_size = frames.shape[-2:]
    frames, resize_size = process_frames(frames, h, w)
    target_size = (h, w)
    
    # Load and adjust bounding boxes if file exists
    if os.path.exists(bbox_path):
        bboxes = process_bbox(bbox_path, original_size, resize_size, target_size, len(frames))
    else:
        bboxes = []

    return frames.to(device), bboxes


def save_video(frames: torch.Tensor, path, frame_ids=None, save_frame=False):
    """Save video frames (optionally as individual images)."""
    os.makedirs(path, exist_ok=True)
    if frame_ids is None:
        frame_ids = [i for i in range(len(frames))]
    frames = frames[frame_ids]

    if save_frame:
        frame_dir = os.path.join(path, "frames")
        save_frames(frames, frame_dir, frame_ids=frame_ids)
        return frame_dir

def save_frames(frames: torch.Tensor, path, ext="png", frame_ids=None):
    """Save tensor frames as individual image files."""
    os.makedirs(path, exist_ok=True)
    if frame_ids is None:
        frame_ids = [i for i in range(len(frames))]
    for i, frame in zip(frame_ids, frames):
        T.ToPILImage()(frame).save(os.path.join(path, '{:04}.{}'.format(i, ext)))


def load_latent(latent_path, t, frame_ids=None, load_all_latents=False):
    """Load noisy latents for a specific timestep or all steps."""
    if load_all_latents:
        latent_fname = f'noisy_latents_all_steps.pt'
    else:
        latent_fname = f'noisy_latents_{t}.pt'

    lp = os.path.join(latent_path, latent_fname)
    assert os.path.exists(lp), f"Latent at timestep {t} not found in {latent_path}."

    latents = torch.load(lp)

    if frame_ids is not None:
        if load_all_latents:
            latents = latents[:, frame_ids]
        else:
            latents = latents[frame_ids]
    
    return latents


@torch.no_grad()
def prepare_depth(pipe, frames, frame_ids, work_dir):
    """Prepare and cache depth maps for frames using the depth estimator."""
    depth_ls = []
    depth_dir = os.path.join(work_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    for frame, frame_id in zip(frames, frame_ids):
        depth_path = os.path.join(depth_dir, "{:04}.pt".format(frame_id))
        depth = load_depth(pipe, depth_path, frame)
        depth_ls += [depth]
    print(f"[INFO] loaded depth images from {depth_path}")
    return torch.cat(depth_ls)


def load_depth(model, depth_path, input_image, dtype=torch.float32):
    """Load or compute and cache depth map for a single frame."""
    if os.path.exists(depth_path):
        depth_map = torch.load(depth_path)
    else:
        input_image = T.ToPILImage()(input_image.squeeze())
        depth_map = prepare_depth_map(model, input_image, dtype=dtype, device=model.device)
        torch.save(depth_map, depth_path)
        # Save visualization
        depth_image = (((depth_map + 1.0) / 2.0) * 255).to(torch.uint8)
        T.ToPILImage()(depth_image.squeeze()).convert("L").save(depth_path.replace(".pt", ".png"))

    return depth_map


@torch.no_grad()
def prepare_depth_map(model, image, depth_map=None, batch_size=1, do_classifier_free_guidance=False, dtype=torch.float32, device="cuda"):
    """Compute depth map from input image using model's depth estimator."""
    if isinstance(image, Image.Image):
        image = [image]
    else:
        image = list(image)

    if isinstance(image[0], Image.Image):
        width, height = image[0].size
    elif isinstance(image[0], np.ndarray):
        width, height = image[0].shape[:-1]
    else:
        height, width = image[0].shape[-2:]

    if depth_map is None:
        pixel_values = model.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device)
        context_manager = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
        with context_manager:
            ret = model.depth_estimator(pixel_values)
            depth_map = ret.predicted_depth

    depth_map = depth_map.to(device=device, dtype=dtype)

    # Handle background (invalid depth values)
    indices = depth_map != -1
    bg_indices = depth_map == -1
    min_d = depth_map[indices].min()
    if bg_indices.sum() > 0:
        depth_map[bg_indices] = min_d - 10

    # Resize to latent space resolution
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(height // model.vae_scale_factor, width // model.vae_scale_factor),
        mode="bicubic",
        align_corners=False,
    )

    # Normalize to [-1, 1]
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
    depth_map = depth_map.to(dtype)

    # Repeat for batch size and classifier-free guidance if needed
    if depth_map.shape[0] < batch_size:
        repeat_by = batch_size // depth_map.shape[0]
        depth_map = depth_map.repeat(repeat_by, 1, 1, 1)

    depth_map = torch.cat([depth_map] * 2) if do_classifier_free_guidance else depth_map
    return depth_map


def get_latents_dir(latents_path, model_key):
    """Generate subdirectory path for saved latents based on model name."""
    model_key = model_key.split("/")[-1]
    return os.path.join(latents_path, model_key)


def get_controlnet_kwargs(controlnet, x, cond, t, controlnet_cond, controlnet_scale=1.0):
    """Compute additional residuals from ControlNet for UNet conditioning."""
    down_block_res_samples, mid_block_res_sample = controlnet(
        x,
        t,
        encoder_hidden_states=cond,
        controlnet_cond=controlnet_cond,
        return_dict=False,
    )
    # Scale residuals
    down_block_res_samples = [down_block_res_sample * controlnet_scale for down_block_res_sample in down_block_res_samples]
    mid_block_res_sample *= controlnet_scale

    controlnet_kwargs = {
        "down_block_additional_residuals": down_block_res_samples,
        "mid_block_additional_residual": mid_block_res_sample
    }
    return controlnet_kwargs 


def get_frame_ids(frame_range, frame_ids=None):
    """Parse and display selected frame indices."""
    if frame_ids is None:
        frame_ids = list(range(*frame_range))
    frame_ids = sorted(frame_ids)

    if len(frame_ids) > 4:
        frame_ids_str = "{} {} ... {} {}".format(*frame_ids[:2], *frame_ids[-2:])
    else:
        frame_ids_str = " ".join(["{}"] * len(frame_ids)).format(*frame_ids)
    print("[INFO] frame indexes: ", frame_ids_str)
    return frame_ids


def prepare_control(control, frames, frame_ids, save_path):
    """Preprocess and cache ControlNet conditioning images if needed."""
    if control not in CONTROLNET_DICT.keys():
        print(f"[WARNING] unknown controlnet type {control}")
        return None

    control_subdir = f'{save_path}/{control}_image'

    preprocess_flag = True
    if os.path.exists(control_subdir):
        print(f"[INFO] load control image from {control_subdir}.")
        control_image_ls = []
        for frame_id in frame_ids:
            image_path = os.path.join(control_subdir, "{:04}.png".format(frame_id))
            if not os.path.exists(image_path):
                break
            control_image_ls += [load_image(image_path)]
        else:
            preprocess_flag = False
            control_images = torch.cat(control_image_ls)

    if preprocess_flag:
        print("[INFO] preprocessing control images...")
        control_images = control_preprocess(frames, control)
        print(f"[INFO] save control images to {control_subdir}.")
        os.makedirs(control_subdir, exist_ok=True)
        for image, frame_id in zip(control_images, frame_ids):
            image_path = os.path.join(control_subdir, "{:04}.png".format(frame_id))
            T.ToPILImage()(image).save(image_path)

    return control_images


def scale_proportion(obj_box, H, W, use_legacy=False):
    """Scale normalized bounding box [0,1] to pixel coordinates in (H, W) image."""
    if use_legacy:
        x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
    else:
        # Round width/height separately to preserve box size invariance
        x_min, y_min = round(obj_box[0] * W), round(obj_box[1] * H)
        box_w, box_h = round((obj_box[2] - obj_box[0]) * W), round((obj_box[3] - obj_box[1]) * H)
        x_max, y_max = x_min + box_w, y_min + box_h
        
        # Clamp to image bounds
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, W), min(y_max, H)
        
    return x_min, y_min, x_max, y_max


def free_memory():
    """Free GPU memory by collecting garbage and clearing CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
