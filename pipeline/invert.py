import torch.nn as nn
import torch
from tqdm import tqdm
import os
from transformers import logging

from utils import load_config, save_config
from utils import get_controlnet_kwargs, get_latents_dir, init_model, seed_everything
from utils import load_video, prepare_depth, save_frames, control_preprocess

logging.set_verbosity_error()


class Inverter(nn.Module):
    """Core module for performing DDIM inversion on video frames using Stable Diffusion."""
    def __init__(self, pipe, scheduler, config):
        super().__init__()

        self.device = config.device
        # Use depth conditioning if the model is a depth-based variant
        self.use_depth = config.sd_version == "depth"
        self.model_key = config.model_key

        self.config = config
        inv_config = config.inversion

        # Set floating-point precision (fp16 or fp32)
        float_precision = inv_config.float_precision if "float_precision" in inv_config else config.float_precision
        if float_precision == "fp16":
            self.dtype = torch.float16
            print("[INFO] float precision fp16. Use torch.float16.")
        else:
            self.dtype = torch.float32
            print("[INFO] float precision fp32. Use torch.float32.")

        # Components from the diffusion pipeline
        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder

        # Enable xformers for memory-efficient attention if configured
        if config.enable_xformers_memory_efficient_attention:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ModuleNotFoundError:
                print("[WARNING] xformers not found. Disable xformers attention.")

        # ControlNet settings
        self.control = inv_config.control
        if self.control != "none":
            self.controlnet = pipe.controlnet

        self.controlnet_scale = inv_config.control_scale

        # Scheduler configuration for saving intermediate latents
        scheduler.set_timesteps(inv_config.save_steps)
        self.timesteps_to_save = scheduler.timesteps
        scheduler.set_timesteps(inv_config.steps)

        self.scheduler = scheduler

        # Inversion hyperparameters
        self.prompt = inv_config.prompt
        self.recon = inv_config.recon
        self.save_latents = inv_config.save_intermediate
        self.use_blip = inv_config.use_blip
        self.steps = inv_config.steps
        self.batch_size = inv_config.batch_size
        self.force = inv_config.force

        # Video/frame settings
        self.n_frames = inv_config.n_frames
        self.frame_height, self.frame_width = config.height, config.width
        self.work_dir = config.work_dir


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt=None, device="cuda"):
        """Encode text prompt into embeddings using the text encoder."""
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        if negative_prompt is not None:
            uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                          return_tensors='pt')
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode VAE latents to image tensors in [0, 1] range."""
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs


    @torch.no_grad()
    def decode_latents_batch(self, latents):
        """Decode latents in batches to manage memory."""
        imgs = []
        batch_latents = latents.split(self.batch_size, dim=0)
        for latent in batch_latents:
            imgs += [self.decode_latents(latent)]
        imgs = torch.cat(imgs)
        return imgs


    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode images to VAE latents."""
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist 
            latents = posterior.mean * 0.18215
        return latents


    @torch.no_grad()
    def encode_imgs_batch(self, imgs):
        """Encode images to latents in batches to manage memory."""
        latents = []
        batch_imgs = imgs.split(self.batch_size, dim=0)
        for img in batch_imgs: 
            latents += [self.encode_imgs(img)] 
        latents = torch.cat(latents)  # [total_n, 4, 64, 64]
        return latents


    @torch.no_grad()
    def ddim_inversion(self, x, conds, save_path):
        """Perform DDIM inversion to obtain noisy latents from clean latents."""
        print("[INFO] start DDIM Inversion!")
        timesteps = reversed(self.scheduler.timesteps)
        total_latents = []  # Collect latents at every timestep

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for i, t in enumerate(tqdm(timesteps)):
                noises = []
                x_index = torch.arange(len(x))  # Indices for batching
                batches = x_index.split(self.batch_size, dim=0)
                for batch in batches: 
                    noise = self.pred_noise(
                        x[batch], conds[batch], timesteps[i], batch_idx=batch)
                    noises += [noise]
                noises = torch.cat(noises)
                x = self.pred_next_x(x, noises, t, i, inversion=True)
                x_to_save = x.detach().cpu()  
                total_latents.append(x_to_save)
                if self.save_latents and t in self.timesteps_to_save:
                    torch.save(x, os.path.join(
                        save_path, f'noisy_latents_{t}.pt'))

        # Save all intermediate latents
        total_latents = torch.stack(total_latents, dim=0)
        total_latents_path = os.path.join(save_path, "noisy_latents_all_steps.pt")
        torch.save(total_latents, total_latents_path)
        print(f"[INFO] all timestep latents saved to: {total_latents_path}")

        # Save final inverted latent
        pth = os.path.join(save_path, f'noisy_latents_{t}.pt')
        torch.save(x, pth)
        print(f"[INFO] inverted latent saved to: {pth}")
        return x


    @torch.no_grad()
    def ddim_sample(self, x, conds):
        """Perform DDIM sampling (denoising) to reconstruct frames from inverted latents."""
        print("[INFO] reconstructing frames...")
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for i, t in enumerate(tqdm(timesteps)):
                noises = []
                x_index = torch.arange(len(x))
                batches = x_index.split(self.batch_size, dim=0)
                for batch in batches:
                    noise = self.pred_noise(
                        x[batch], conds[batch], t, batch_idx=batch)
                    noises += [noise]
                noises = torch.cat(noises)
                x = self.pred_next_x(x, noises, t, i, inversion=False)
        return x


    @torch.no_grad()
    def pred_noise(self, x, cond, t, batch_idx=None):
        """Predict noise using UNet, optionally with depth or ControlNet conditioning."""
        # Depth conditioning for depth-based models
        if self.use_depth:
            depth = self.depths
            if batch_idx is not None:
                depth = depth[batch_idx]
            x = torch.cat([x, depth.to(x)], dim=1)

        kwargs = {}
        # ControlNet conditioning if enabled
        if self.control != "none":
            if batch_idx is None:
                controlnet_cond = self.controlnet_images
            else:
                controlnet_cond = self.controlnet_images[batch_idx]
            controlnet_kwargs = get_controlnet_kwargs(self.controlnet, x, cond, t, controlnet_cond, self.controlnet_scale)
            kwargs.update(controlnet_kwargs)

        # UNet forward pass to predict noise
        eps = self.unet(x, t, encoder_hidden_states=cond, **kwargs).sample
        return eps


    @torch.no_grad()
    def pred_next_x(self, x, eps, t, i, inversion=False):
        """Compute next latent based on DDIM formula (forward for inversion, reverse for sampling)."""
        if inversion:
            timesteps = reversed(self.scheduler.timesteps)
        else:
            timesteps = self.scheduler.timesteps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        if inversion:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )
        else:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else self.scheduler.final_alpha_cumprod
            )
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        if inversion:
            pred_x0 = (x - sigma_prev * eps) / mu_prev
            x = mu * pred_x0 + sigma * eps
        else:
            pred_x0 = (x - sigma * eps) / mu
            x = mu_prev * pred_x0 + sigma_prev * eps

        return x


    @torch.no_grad()
    def prepare_cond(self, prompts, n_frames):
        """Prepare text conditioning embeddings for all frames."""
        if isinstance(prompts, str):
            prompts = [prompts] * n_frames
            cond = self.get_text_embeds(prompts[0])
            conds = torch.cat([cond] * n_frames)
        elif isinstance(prompts, list):
            cond_ls = []
            for prompt in prompts:
                cond = self.get_text_embeds(prompt)
                cond_ls += [cond]
            conds = torch.cat(cond_ls)
        return conds, prompts


    def check_latent_exists(self, save_path):
        """Check if required inverted latents already exist."""
        save_timesteps = [self.scheduler.timesteps[0]]
        if self.save_latents:
            save_timesteps += self.timesteps_to_save
        for ts in save_timesteps:
            latent_path = os.path.join(
                save_path, f'noisy_latents_{ts}.pt')
            if not os.path.exists(latent_path):
                return False
        return True


    @torch.no_grad()
    def __call__(self, data_path, bbox_path, save_path):
        """Main inversion pipeline entry point."""
        self.scheduler.set_timesteps(self.steps)
        save_path = get_latents_dir(save_path, self.model_key)
        os.makedirs(save_path, exist_ok=True)

        # Skip inversion if latents exist and force is not enabled
        if self.check_latent_exists(save_path) and not self.force:
            print(f"[INFO] inverted latents exist at: {save_path}. Skip inversion! Set 'inversion.force: True' to invert again.")
            return

        # Load and preprocess video frames
        frames, bbox = load_video(data_path, bbox_path, self.frame_height, self.frame_width, device=self.device)

        frame_ids = list(range(len(frames)))
        
        if self.n_frames is not None:
            frame_ids = frame_ids[:self.n_frames]
        frames = frames[frame_ids]

        # Prepare depth maps if using depth model
        if self.use_depth:
            self.depths = prepare_depth(self.pipe, frames, frame_ids, self.work_dir)

        # Prepare text conditioning
        conds, prompts = self.prepare_cond(self.prompt, len(frames))
        with open(os.path.join(save_path, 'inversion_prompts.txt'), 'w') as f:
            f.write('\n'.join(prompts))

        # Prepare ControlNet conditioning images if enabled
        if self.control != "none":
            images = control_preprocess(
                frames, self.control)
            self.controlnet_images = images.to(self.device)

        # Encode frames to clean latents
        latents = self.encode_imgs_batch(frames)
        torch.cuda.empty_cache()
        print(f"[INFO] clean latents shape: {latents.shape}")

        # Perform DDIM inversion
        inverted_x = self.ddim_inversion(latents, conds, save_path) 
        save_config(self.config, save_path, inv=True)

        # Optional reconstruction to verify inversion quality
        if self.recon:
            latent_reconstruction = self.ddim_sample(inverted_x, conds)

            torch.cuda.empty_cache()
            recon_frames = self.decode_latents_batch(
                latent_reconstruction)

            recon_save_path = os.path.join(save_path, 'recon_frames')
            save_frames(recon_frames, recon_save_path, frame_ids=frame_ids)


if __name__ == "__main__":
    # Load configuration and initialize model
    config = load_config()
    pipe, scheduler, model_key = init_model(
        config.device, config.sd_version, config.model_key, config.inversion.control, config.float_precision)
    config.model_key = model_key
    seed_everything(config.seed)

    # Create inverter and run on input data
    inversion = Inverter(pipe, scheduler, config)
    inversion(config.input_path, config.inversion.save_path)