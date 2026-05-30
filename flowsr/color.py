from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and standard deviation of a 4D (B, C, H, W) tensor."""
    b, c = feat.shape[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def _adaptive_instance_normalization(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """Rescale ``content`` to match the per-channel statistics of ``style``."""
    size = content.size()
    style_mean, style_std = _calc_mean_std(style)
    content_mean, content_std = _calc_mean_std(content)
    normalized = (content - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _wavelet_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply a single level of wavelet (low-pass) blur to a (1, 3, H, W) tensor."""
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(image, kernel, groups=3, dilation=radius)


def _wavelet_decomposition(image: torch.Tensor, levels: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an image into accumulated high-frequency detail and a low-frequency residual."""
    high_freq = torch.zeros_like(image)
    low_freq = image
    for i in range(levels):
        radius = 2**i
        low_freq = _wavelet_blur(image, radius)
        high_freq = high_freq + (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def _wavelet_reconstruction(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """Combine the high frequencies of ``content`` with the low frequencies of ``style``."""
    content_high_freq, _ = _wavelet_decomposition(content)
    _, style_low_freq = _wavelet_decomposition(style)
    return content_high_freq + style_low_freq


def adain_color_fix(output: Image.Image, reference: Image.Image) -> Image.Image:
    """Match ``output`` channel statistics to ``reference`` via adaptive instance normalization."""
    output_tensor = _pil_to_tensor(output)
    reference_tensor = _pil_to_tensor(reference.resize(output.size, Image.Resampling.BICUBIC))
    result = _adaptive_instance_normalization(output_tensor, reference_tensor)
    return _tensor_to_pil(result)


def wavelet_color_fix(output: Image.Image, reference: Image.Image) -> Image.Image:
    """Keep the detail of ``output`` while borrowing the low-frequency color of ``reference``."""
    output_tensor = _pil_to_tensor(output)
    reference_tensor = _pil_to_tensor(reference.resize(output.size, Image.Resampling.BICUBIC))
    result = _wavelet_reconstruction(output_tensor, reference_tensor)
    return _tensor_to_pil(result)
