#!/usr/bin/env python3
"""Run a Pi3-compatible checkpoint on the frozen 43-subset benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=package_root)
    parser.add_argument("--pi3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-width", type=int, default=518)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, path: Path, strict: bool) -> None:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_model

        load_model(model, str(path), strict=strict)
        return
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict:
        print(
            f"Loaded non-strictly: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )


def load_images(paths: list[Path], width: int) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in paths]
    original_width, original_height = images[0].size
    height = max(14, round(original_height * (width / original_width) / 14) * 14)
    arrays = []
    for image in images:
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        arrays.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(arrays).unsqueeze(0)


def ordered_image_paths(subset_dir: Path) -> list[Path]:
    with (subset_dir / "ordered_images.csv").open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["index"]))
    if len(rows) != 20 or [int(row["index"]) for row in rows] != list(range(20)):
        raise ValueError(f"Invalid 20-frame order in {subset_dir}")
    return [subset_dir / row["path"] for row in rows]


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.pi3_repo.resolve()))
    from pi3.models.pi3 import Pi3

    model = Pi3()
    load_checkpoint(model, args.checkpoint.resolve(), strict=not args.non_strict)
    model = model.to(args.device).eval()
    manifest = json.loads((args.benchmark_root / "benchmark.json").read_text())
    failures = []
    for number, subset in enumerate(manifest["subsets"], start=1):
        scene = subset["scene"]
        subset_id = subset["subset_id"]
        output_dir = args.output_dir / scene / subset_id
        output_path = output_dir / "pred_c2w.npy"
        if output_path.is_file() and not args.overwrite:
            print(f"[{number}/43] skip {scene}/{subset_id}")
            continue
        try:
            subset_dir = args.benchmark_root / subset["path"]
            paths = ordered_image_paths(subset_dir)
            inputs = load_images(paths, args.image_width).to(args.device)
            is_cuda = str(args.device).startswith("cuda")
            dtype = torch.bfloat16 if is_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            start = time.time()
            with torch.inference_mode(), torch.amp.autocast(args.device, dtype=dtype, enabled=is_cuda):
                prediction = model.inference(inputs)
            poses = prediction["camera_poses"][0].detach().float().cpu().numpy()
            if poses.shape != (20, 4, 4):
                raise ValueError(f"Expected camera poses (20, 4, 4), got {poses.shape}")
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_path, poses)
            metadata = {
                "scene": scene,
                "subset_id": subset_id,
                "num_images": 20,
                "checkpoint": str(args.checkpoint.resolve()),
                "image_width": args.image_width,
                "elapsed_seconds": time.time() - start,
            }
            (output_dir / "inference.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"[{number}/43] {scene}/{subset_id}: {metadata['elapsed_seconds']:.2f}s")
        except Exception as error:
            failures.append({"scene": scene, "subset_id": subset_id, "error": repr(error)})
            if not args.continue_on_error:
                raise
            print(f"[{number}/43] FAILED {scene}/{subset_id}: {error}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} subsets failed; see {args.output_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
