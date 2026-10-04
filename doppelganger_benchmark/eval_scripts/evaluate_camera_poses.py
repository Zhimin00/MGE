#!/usr/bin/env python3
"""Evaluate Pi3 camera-to-world predictions with scale-free relative-pose metrics."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = (5, 10, 15, 30)
GROUPS = ("all", "reference", "cross", "doppelganger")


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=package_root)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected (N,3,4) or (N,4,4), got {poses.shape}")
    if poses.shape[1:] == (4, 4):
        return poses
    bottom = np.broadcast_to(np.asarray([0.0, 0.0, 0.0, 1.0]), (len(poses), 1, 4))
    return np.concatenate((poses, bottom), axis=1)


def pair_errors(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> list[dict]:
    pred_c2w, gt_c2w = homogeneous(pred_c2w), homogeneous(gt_c2w)
    if pred_c2w.shape != gt_c2w.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred_c2w.shape} vs {gt_c2w.shape}")
    pred_w2c, gt_w2c = np.linalg.inv(pred_c2w), np.linalg.inv(gt_c2w)
    rows = []
    for left, right in combinations(range(len(pred_c2w)), 2):
        pred_relative = pred_w2c[left] @ pred_c2w[right]
        gt_relative = gt_w2c[left] @ gt_c2w[right]
        delta = pred_relative[:3, :3] @ gt_relative[:3, :3].T
        rotation = float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))))
        pred_t, gt_t = pred_relative[:3, 3], gt_relative[:3, 3]
        denominator = np.linalg.norm(pred_t) * np.linalg.norm(gt_t)
        translation = 90.0 if denominator <= 1e-12 else float(
            np.degrees(np.arccos(np.clip(abs(float(np.dot(pred_t, gt_t) / denominator)), 0.0, 1.0)))
        )
        rows.append({"left": left, "right": right, "rotation": rotation, "translation": translation})
    return rows


def split_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups = {name: [] for name in GROUPS}
    groups["all"] = rows
    for row in rows:
        left_ref, right_ref = row["left"] < 15, row["right"] < 15
        if left_ref and right_ref:
            groups["reference"].append(row)
        elif left_ref != right_ref:
            groups["cross"].append(row)
        else:
            groups["doppelganger"].append(row)
    return groups


def auc(errors: np.ndarray, threshold: int) -> float:
    histogram, _ = np.histogram(errors, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(float) / len(errors))))


def metrics(rows: list[dict]) -> dict[str, float]:
    rotation = np.asarray([row["rotation"] for row in rows])
    translation = np.asarray([row["translation"] for row in rows])
    joint = np.maximum(rotation, translation)
    result = {
        "num_pairs": len(rows),
        "rotation_error_deg_mean": float(rotation.mean()),
        "translation_error_deg_mean": float(translation.mean()),
    }
    for threshold in THRESHOLDS:
        result[f"auc_at_{threshold}"] = auc(joint, threshold)
        result[f"rra_at_{threshold}"] = float(np.mean(rotation < threshold))
        result[f"rta_at_{threshold}"] = float(np.mean(translation < threshold))
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.benchmark_root / "benchmark.json").read_text())
    rows = []
    for subset in manifest["subsets"]:
        scene, subset_id = subset["scene"], subset["subset_id"]
        subset_dir = args.benchmark_root / subset["path"]
        prediction = args.predictions_root / scene / subset_id / "pred_c2w.npy"
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        row = {"scene": scene, "subset_id": subset_id}
        for group, errors in split_groups(pair_errors(np.load(prediction), np.load(subset_dir / "gt_c2w.npy"))).items():
            row.update({f"{group}_{key}": value for key, value in metrics(errors).items()})
        rows.append(row)
    subset_frame = pd.DataFrame(rows)
    numeric = [column for column in subset_frame.select_dtypes(include=[np.number]).columns]
    overall = pd.DataFrame([{"subsets": len(subset_frame), **subset_frame[numeric].mean().to_dict()}])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_frame.to_csv(args.output_dir / "subset_metrics.csv", index=False)
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    print(f"Evaluated {len(subset_frame)} subsets; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
