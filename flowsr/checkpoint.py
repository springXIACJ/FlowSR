"""Load FlowSR safetensors checkpoints.

The supported (and only) inference checkpoint format is safetensors
(``checkpoints/flowsr.safetensors``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

SAFETENSORS_METADATA_KEY = "flowsr_checkpoint_metadata"
CANONICAL_TENSOR_GROUPS = ("unet_state_dict", "vae_state_dict")
CANONICAL_METADATA_FIELDS = (
    "unet_rank",
    "unet_lora_target_modules",
    "vae_rank",
    "vae_lora_target_modules",
)


class CheckpointConversionError(RuntimeError):
    """Raised when a FlowSR checkpoint cannot be loaded or converted."""


def load_flowsr_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a FlowSR safetensors checkpoint into the canonical dict layout."""
    checkpoint_path = Path(path)
    if checkpoint_path.suffix.lower() == ".safetensors":
        return load_flowsr_checkpoint_safetensors(checkpoint_path)
    # Optional legacy converter for original PyTorch checkpoints; absent from
    # safetensors-only builds, in which case we surface a clear error.
    try:
        from flowsr.legacy import load_legacy_checkpoint
    except ModuleNotFoundError as exc:
        raise CheckpointConversionError(
            f"Unsupported checkpoint format '{checkpoint_path.suffix or checkpoint_path.name}'. "
            "Only safetensors checkpoints are supported (checkpoints/flowsr.safetensors)."
        ) from exc
    return load_legacy_checkpoint(checkpoint_path)


def load_flowsr_checkpoint_safetensors(path: Path | str) -> dict[str, Any]:
    checkpoint_path = Path(path)
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        raw_metadata = metadata.get(SAFETENSORS_METADATA_KEY)
        if raw_metadata is None:
            raise CheckpointConversionError(
                f"Safetensors checkpoint {checkpoint_path} is missing FlowSR metadata."
            )
        payload = json.loads(raw_metadata)
        checkpoint = dict(payload.get("fields", {}))
        tensor_names = payload.get("tensor_names", {})
        for group in CANONICAL_TENSOR_GROUPS:
            restored = {}
            for tensor_key, original_key in tensor_names.get(group, {}).items():
                restored[original_key] = handle.get_tensor(tensor_key)
            if restored:
                checkpoint[group] = restored
    return checkpoint
