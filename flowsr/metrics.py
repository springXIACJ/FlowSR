from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from flowsr.infer import IMAGE_EXTENSIONS

PAIRED_METRICS = {"psnr", "ssim", "lpips", "dists"}
NO_REFERENCE_METRICS = {"niqe", "musiq", "maniqa-pipal", "clipiqa"}
METRIC_ALIASES = {
    "maniqa": "maniqa-pipal",
    "maniqa_pipal": "maniqa-pipal",
}
# Reporting order, display label (with the better-direction arrow), and the number
# of decimal places to print, e.g. "25.54 0.7434 0.2728 0.2013 112.60 5.28 69.22 0.6486 0.6701".
METRIC_DISPLAY = (
    ("psnr", "PSNR ↑", 2),
    ("ssim", "SSIM ↑", 4),
    ("lpips", "LPIPS ↓", 4),
    ("dists", "DISTS ↓", 4),
    ("fid", "FID ↓", 2),
    ("niqe", "NIQE ↓", 2),
    ("musiq", "MUSIQ ↑", 2),
    ("maniqa-pipal", "MANIQA ↑", 4),
    ("clipiqa", "CLIPIQA ↑", 4),
)
DEFAULT_METRICS = [name for name, _, _ in METRIC_DISPLAY]


@dataclass(frozen=True)
class ImagePair:
    name: str
    sr_path: Path
    gt_path: Path


@dataclass(frozen=True)
class MetricPlan:
    paired: list[str]
    no_reference: list[str]
    fid: bool


def collect_image_pairs(sr_dir: Path | str, gt_dir: Path | str, recursive: bool = False) -> list[ImagePair]:
    sr_root = Path(sr_dir)
    gt_root = Path(gt_dir)
    if not sr_root.is_dir():
        raise ValueError(f"SR path is not a directory: {sr_root}")
    if not gt_root.is_dir():
        raise ValueError(f"GT path is not a directory: {gt_root}")

    sr_images = _index_images(sr_root, recursive=recursive)
    gt_images = _index_images(gt_root, recursive=recursive)
    missing_gt = sorted(set(sr_images).difference(gt_images))
    missing_sr = sorted(set(gt_images).difference(sr_images))
    if missing_gt or missing_sr:
        parts = []
        if missing_gt:
            parts.append(f"missing GT for: {', '.join(missing_gt[:8])}")
        if missing_sr:
            parts.append(f"missing SR for: {', '.join(missing_sr[:8])}")
        raise ValueError("; ".join(parts))

    return [
        ImagePair(name=name, sr_path=sr_images[name], gt_path=gt_images[name])
        for name in sorted(sr_images)
    ]


def split_metric_names(metric_names: Iterable[str]) -> MetricPlan:
    paired: list[str] = []
    no_reference: list[str] = []
    use_fid = False

    for raw_name in metric_names:
        name = METRIC_ALIASES.get(raw_name.lower(), raw_name.lower())
        if name in PAIRED_METRICS:
            paired.append(name)
        elif name in NO_REFERENCE_METRICS:
            no_reference.append(name)
        elif name == "fid":
            use_fid = True
        else:
            known = sorted(PAIRED_METRICS | NO_REFERENCE_METRICS | {"fid"} | set(METRIC_ALIASES))
            raise ValueError(f"Unknown metric '{raw_name}'. Known metrics: {', '.join(known)}")

    return MetricPlan(paired=paired, no_reference=no_reference, fid=use_fid)


def summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        key: round(sum(row[key] for row in rows) / len(rows), 6)
        for key in keys
    }


def format_summary_table(summary: dict[str, float]) -> tuple[str, str]:
    """Return aligned (header, values) lines in the canonical reporting order."""
    headers: list[str] = []
    values: list[str] = []
    for name, label, decimals in METRIC_DISPLAY:
        if name not in summary:
            continue
        value = f"{summary[name]:.{decimals}f}"
        width = max(len(label), len(value))
        headers.append(label.ljust(width))
        values.append(value.ljust(width))
    return "  ".join(headers).rstrip(), "  ".join(values).rstrip()


def evaluate_directories(
    sr_dir: Path,
    gt_dir: Path,
    metrics: Iterable[str],
    device: str,
    recursive: bool = False,
) -> dict:
    try:
        import pyiqa
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Metric evaluation requires optional dependencies. Install them with: "
            'uv pip install -e ".[metrics]"'
        ) from exc

    plan = split_metric_names(metrics)
    pairs = collect_image_pairs(sr_dir, gt_dir, recursive=recursive)
    resolved_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[{Path(sr_dir).name}] evaluating {len(pairs)} image pair(s) on {resolved_device}\n")

    paired_metrics = {name: _create_metric(pyiqa, name, resolved_device) for name in plan.paired}
    no_ref_metrics = {name: _create_metric(pyiqa, name, resolved_device) for name in plan.no_reference}

    rows = []
    for pair in tqdm(pairs, desc=f"metrics:{Path(sr_dir).name}", unit="img"):
        start = time.time()
        scores = {}
        with torch.no_grad():
            for name, metric in paired_metrics.items():
                scores[name] = _to_float(metric(str(pair.sr_path), str(pair.gt_path)))
            for name, metric in no_ref_metrics.items():
                scores[name] = _to_float(metric(str(pair.sr_path)))
        rows.append({
            "name": pair.name,
            "sr_path": str(pair.sr_path),
            "gt_path": str(pair.gt_path),
            "runtime_sec": round(time.time() - start, 4),
            "scores": scores,
        })

    summary = summarize_metric_rows([row["scores"] for row in rows])
    if plan.fid:
        fid_metric = pyiqa.create_metric("fid", device=resolved_device)
        summary["fid"] = _to_float(fid_metric(str(gt_dir), str(sr_dir)))

    return {
        "sr_dir": str(sr_dir),
        "gt_dir": str(gt_dir),
        "num_images": len(pairs),
        "metrics": list(summary.keys()),
        "summary": summary,
        "images": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SR outputs against GT images with pyiqa metrics.")
    parser.add_argument("--sr", type=Path, nargs="+", required=True, help="SR output directories.")
    parser.add_argument("--gt", type=Path, nargs="+", required=True, help="Ground-truth image directories.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to compute.")
    parser.add_argument("--output-dir", type=Path, default=Path("metrics"), help="Directory for logs and JSON.")
    parser.add_argument("--log-name", default="flowsr_metrics", help="Base name for output files.")
    parser.add_argument("--device", default="auto", help="Device for pyiqa metrics: auto, cuda, or cpu.")
    parser.add_argument("--recursive", action="store_true", help="Match images recursively by relative stem.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.sr) != len(args.gt):
        parser.error("--sr and --gt must contain the same number of directories")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(args.output_dir, args.log_name)
    logger.info("Starting FlowSR metric evaluation")
    logger.info("SR directories: %s", ", ".join(str(path) for path in args.sr))
    logger.info("GT directories: %s", ", ".join(str(path) for path in args.gt))
    logger.info("Metrics: %s", ", ".join(args.metrics))

    try:
        results = [
            evaluate_directories(sr_dir, gt_dir, args.metrics, args.device, recursive=args.recursive)
            for sr_dir, gt_dir in zip(args.sr, args.gt)
        ]
    except Exception as exc:
        logger.error("%s", exc)
        print(str(exc), file=sys.stderr)
        return 2

    for result in results:
        name = Path(result["sr_dir"]).name
        header, values = format_summary_table(result["summary"])
        logger.info("[%s] %d images", name, result["num_images"])
        logger.info("[%s] %s", name, header)
        logger.info("[%s] %s", name, values)

    output_path = args.output_dir / f"{args.log_name}.json"
    output_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    logger.info("Wrote JSON results to %s\n", output_path)
    return 0


def _index_images(root: Path, recursive: bool) -> dict[str, Path]:
    pattern = "**/*" if recursive else "*"
    images: dict[str, Path] = {}
    for path in root.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = str(path.relative_to(root).with_suffix("")) if recursive else path.stem
        if key in images:
            raise ValueError(f"Duplicate image key '{key}' in {root}")
        images[key] = path
    if not images:
        raise ValueError(f"No supported images found in: {root}")
    return images


def _create_metric(pyiqa, name: str, device):
    # PSNR/SSIM are reported on the Y channel in YCbCr, matching common SR benchmarks.
    if name in {"psnr", "ssim"}:
        return pyiqa.create_metric(name, test_y_channel=True, color_space="ycbcr").to(device)
    return pyiqa.create_metric(name, device=device)


def _to_float(value) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return round(float(value), 6)


def _setup_logger(output_dir: Path, log_name: str) -> logging.Logger:
    logger = logging.getLogger("flowsr.metrics")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(output_dir / f"{log_name}_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


if __name__ == "__main__":
    raise SystemExit(main())
