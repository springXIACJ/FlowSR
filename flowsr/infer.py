from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image

from flowsr.defaults import (
    DEFAULT_BASE_MODEL,
    DEFAULT_FLOW_SCHEDULER_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_POSITIVE_PROMPT,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class CheckpointValidationError(RuntimeError):
    """Raised when a checkpoint cannot be loaded as a PyTorch checkpoint."""


def collect_image_paths(input_path: Path | str) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]

    images = [
        item
        for item in sorted(path.iterdir(), key=lambda p: p.name.lower())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No supported images found in: {path}")
    return images


def make_output_path(image_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{image_path.stem}.png"


def validate_checkpoint(checkpoint: Path | str) -> dict:
    path = Path(checkpoint)
    if not path.exists():
        raise CheckpointValidationError(f"Checkpoint does not exist: {path}")
    try:
        from flowsr.checkpoint import load_flowsr_checkpoint

        loaded = load_flowsr_checkpoint(path)
    except Exception as exc:
        raise CheckpointValidationError(f"Could not load checkpoint {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise CheckpointValidationError(f"Checkpoint {path} did not load as a dictionary.")
    required = {"unet_state_dict", "unet_rank", "unet_lora_target_modules"}
    missing = sorted(required.difference(loaded))
    if missing:
        raise CheckpointValidationError(f"Checkpoint {path} is missing keys: {', '.join(missing)}")
    return {
        "path": str(path),
        "keys": sorted(loaded.keys()),
        "unet_tensors": len(loaded["unet_state_dict"]),
        "vae_tensors": len(loaded.get("vae_state_dict", {})),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FlowSR inference on one image or a folder.")
    parser.add_argument("--input", type=Path, required=False, help="Input image or directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/flowsr.safetensors"),
        help="FlowSR safetensors checkpoint.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--flow-scheduler-model", default=DEFAULT_FLOW_SCHEDULER_MODEL)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--latent-tile-size", type=int, default=96)
    parser.add_argument("--latent-tile-overlap", type=int, default=32)
    parser.add_argument("--align-method", choices=["none", "adain", "wavelet"], default="wavelet")
    parser.add_argument("--positive-prompt", default=DEFAULT_POSITIVE_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--check-checkpoint-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        info = validate_checkpoint(args.checkpoint)
    except CheckpointValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.check_checkpoint_only:
        print(f"\nCheckpoint OK: {info['path']}")
        print(f"UNet tensors: {info['unet_tensors']}")
        print(f"VAE tensors: {info['vae_tensors']}\n")
        return 0

    if args.input is None:
        parser.error("--input is required unless --check-checkpoint-only is set")

    run_inference(args)
    return 0


def run_inference(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from flowsr.color import adain_color_fix, wavelet_color_fix
    from flowsr.model import FlowSRConfig, FlowSRModel

    image_paths = collect_image_paths(args.input)
    print(f"\nFound {len(image_paths)} image(s) to process.")
    print(f"Loading FlowSR model on {args.device} ({args.dtype})...\n")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    model = FlowSRModel(
        FlowSRConfig(
            base_model=args.base_model,
            flow_scheduler_model=args.flow_scheduler_model,
            checkpoint=args.checkpoint,
            device=str(device),
            dtype=dtype,
            latent_tile_size=args.latent_tile_size,
            latent_tile_overlap=args.latent_tile_overlap,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
    )

    print("Model loaded. Running inference...\n")

    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(image_paths, desc="FlowSR", unit="img"):
        image = Image.open(image_path).convert("RGB")
        lr_tensor = _pil_to_tensor(image).to(device=device, dtype=torch.float32)
        _, _, height, width = lr_tensor.shape
        resized = F.interpolate(
            lr_tensor,
            size=(int(height * args.scale), int(width * args.scale)),
            mode="bilinear",
            align_corners=False,
        )
        normalized = resized * 2 - 1
        padded, original_size = _pad_to_multiple(normalized.clamp(-1, 1), multiple=64)
        with torch.no_grad():
            output = model(
                padded.to(dtype=dtype),
                positive_prompt=[args.positive_prompt],
                negative_prompt=[args.negative_prompt],
            )
        output = output[:, :, : original_size[0], : original_size[1]]
        output_image = _tensor_to_pil(output.float().cpu())
        if args.align_method == "adain":
            output_image = adain_color_fix(output_image, image)
        elif args.align_method == "wavelet":
            output_image = wavelet_color_fix(output_image, image)
        output_image.save(make_output_path(image_path, args.output))

    print(f"\nSaved {len(image_paths)} image(s) to {args.output}\n")


def _pil_to_tensor(image: Image.Image):
    import numpy as np
    import torch

    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _tensor_to_pil(tensor) -> Image.Image:
    import numpy as np

    tensor = tensor[0].clamp(-1, 1)
    array = ((tensor * 0.5 + 0.5).permute(1, 2, 0).numpy() * 255).round()
    return Image.fromarray(array.clip(0, 255).astype(np.uint8))


def _pad_to_multiple(tensor, multiple: int):
    import math
    import torch.nn.functional as F

    height, width = tensor.shape[-2:]
    pad_h = math.ceil(height / multiple) * multiple - height
    pad_w = math.ceil(width / multiple) * multiple - width
    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)
    return F.pad(tensor, pad=(0, pad_w, 0, pad_h), mode="reflect"), (height, width)


if __name__ == "__main__":
    raise SystemExit(main())
