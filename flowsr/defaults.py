"""Shared inference defaults.

This module intentionally has no heavy dependencies (torch, diffusers, ...) so it
can be imported from ``infer.py`` without pulling in the full ML stack, preserving
the lazy-import design that keeps arg parsing and the unit tests lightweight.
"""

from __future__ import annotations

# The original "stabilityai/stable-diffusion-2-1-base" repository was removed from
# the Hugging Face Hub. "Manojb/stable-diffusion-2-1-base" is a drop-in re-upload
# of the same weights and is used as the default base model here.
DEFAULT_BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
DEFAULT_FLOW_SCHEDULER_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"

DEFAULT_POSITIVE_PROMPT = (
    "A high-resolution, 8K, ultra-realistic image with sharp focus, "
    "vibrant colors, and natural lighting."
)
DEFAULT_NEGATIVE_PROMPT = (
    "oil painting, cartoon, blur, dirty, messy, low quality, deformation, "
    "low resolution, oversmooth"
)
