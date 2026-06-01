from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

from flowsr.checkpoint import load_flowsr_checkpoint
from flowsr.defaults import DEFAULT_BASE_MODEL, DEFAULT_FLOW_SCHEDULER_MODEL


@dataclass(frozen=True)
class FlowSRConfig:
    base_model: str = DEFAULT_BASE_MODEL
    flow_scheduler_model: str = DEFAULT_FLOW_SCHEDULER_MODEL
    checkpoint: Path = Path("checkpoints/flowsr.safetensors")
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    latent_tile_size: int = 96
    latent_tile_overlap: int = 32
    num_inference_steps: int = 1
    guidance_scale: float = 1.0


def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of timesteps or sigmas can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters)
        if not accepts_timesteps:
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters)
        if not accepts_sigmas:
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FlowSRModel(torch.nn.Module):
    """Inference-only FlowSR model wrapper."""

    def __init__(self, config: FlowSRConfig):
        super().__init__()
        self.config = config
        self.guidance_scale = config.guidance_scale
        self.latent_tile_size = config.latent_tile_size
        self.latent_tile_overlap = config.latent_tile_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.base_model, subfolder="text_encoder", torch_dtype=config.dtype
        ).to(config.device)
        self.vae = AutoencoderKL.from_pretrained(
            config.base_model, subfolder="vae", torch_dtype=config.dtype
        ).to(config.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            config.base_model, subfolder="unet", torch_dtype=config.dtype
        ).to(config.device)

        self._load_lora_checkpoint(config.checkpoint)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.flow_scheduler_model, subfolder="scheduler"
        )
        self.timesteps, _ = retrieve_timesteps(
            self.scheduler, config.num_inference_steps, config.device
        )

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.eval()

    def _load_lora_checkpoint(self, checkpoint: Path) -> None:
        ckpt = load_flowsr_checkpoint(checkpoint)
        if not isinstance(ckpt, dict) or "unet_state_dict" not in ckpt or "unet_rank" not in ckpt:
            raise ValueError("Checkpoint must contain unet_state_dict and unet_rank.")

        if "vae_rank" in ckpt and "vae_state_dict" in ckpt:
            vae_lora_config = LoraConfig(
                r=ckpt["vae_rank"],
                init_lora_weights="gaussian",
                target_modules=ckpt["vae_lora_target_modules"],
            )
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            vae_state = self.vae.state_dict()
            for key, value in ckpt["vae_state_dict"].items():
                new_key = key.replace(".default", ".vae_skip") if ".default" in key else key
                vae_state[new_key] = value
            self.vae.load_state_dict(vae_state)

        unet_lora_config = LoraConfig(
            r=ckpt["unet_rank"],
            init_lora_weights="gaussian",
            target_modules=ckpt["unet_lora_target_modules"],
        )
        self.unet.add_adapter(unet_lora_config)
        unet_state = self.unet.state_dict()
        unet_state.update(ckpt["unet_state_dict"])
        self.unet.load_state_dict(unet_state)

    @torch.no_grad()
    def encode_prompt(self, prompt: list[str]):
        tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.config.device)
        return self.text_encoder(tokens)[0]

    @torch.no_grad()
    def forward(
        self,
        image_tensor: torch.Tensor,
        positive_prompt: list[str],
        negative_prompt: list[str] | None = None,
    ) -> torch.Tensor:
        positive = self.encode_prompt(positive_prompt)
        negative = self.encode_prompt(negative_prompt) if self.guidance_scale > 1 and negative_prompt else None

        latents = self.vae.encode(image_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        for timestep in self.timesteps:
            noise_pred = self._predict_noise(latents, timestep, positive, negative)
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        self.scheduler._step_index = None
        self.scheduler._begin_index = None
        return self.vae.decode(latents / self.vae.config.scaling_factor).sample.clamp(-1, 1)

    def _predict_noise(self, latents, timestep, positive, negative):
        _, _, height, width = latents.shape
        tile_size = min(self.latent_tile_size, height, width)
        if height * width <= tile_size * tile_size:
            return self._predict_noise_tile(latents, timestep, positive, negative)

        overlap = min(self.latent_tile_overlap, max(tile_size - 1, 0))
        weights = self._gaussian_weights(tile_size, tile_size).to(latents.device, dtype=latents.dtype)
        output = torch.zeros_like(latents)
        contributors = torch.zeros_like(latents)

        row_offsets = _tile_offsets(height, tile_size, overlap)
        col_offsets = _tile_offsets(width, tile_size, overlap)
        for y in row_offsets:
            for x in col_offsets:
                tile = latents[:, :, y : y + tile_size, x : x + tile_size]
                pred = self._predict_noise_tile(tile, timestep, positive, negative)
                output[:, :, y : y + tile_size, x : x + tile_size] += pred * weights
                contributors[:, :, y : y + tile_size, x : x + tile_size] += weights
        return output / contributors.clamp_min(1e-6)

    def _predict_noise_tile(self, latents, timestep, positive, negative):
        pos = self.unet(latents, timestep, encoder_hidden_states=positive).sample
        if self.guidance_scale <= 1 or negative is None:
            return pos
        neg = self.unet(latents, timestep, encoder_hidden_states=negative).sample
        return neg + self.guidance_scale * (pos - neg)

    def _gaussian_weights(self, tile_width: int, tile_height: int) -> torch.Tensor:
        var = 0.01
        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [
            np.exp(-((x - midpoint_x) ** 2) / (tile_width * tile_width) / (2 * var))
            for x in range(tile_width)
        ]
        y_probs = [
            np.exp(-((y - midpoint_y) ** 2) / (tile_height * tile_height) / (2 * var))
            for y in range(tile_height)
        ]
        weights = np.outer(y_probs, x_probs).astype(np.float32)
        return torch.tensor(weights).expand(1, self.unet.config.in_channels, tile_height, tile_width)


def _tile_offsets(size: int, tile_size: int, overlap: int) -> list[int]:
    if size <= tile_size:
        return [0]
    stride = tile_size - overlap
    offsets = list(range(0, max(size - tile_size, 0), stride))
    last = size - tile_size
    if not offsets or offsets[-1] != last:
        offsets.append(last)
    return offsets
