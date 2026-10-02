"""Reference-aligned camera-cluster dragging diagnostics."""

from typing import Dict

import numpy as np

from utils.paper_geometry import apply_sim3_to_poses, estimate_sim3


def align_poses_on_reference(pred_c2w: np.ndarray, gt_c2w: np.ndarray, reference_count: int = 15):
    """Robustly align using the lowest-residual 70% of reference centers."""
    pred_centers = pred_c2w[:reference_count, :3, 3]
    gt_centers = gt_c2w[:reference_count, :3, 3]
    indices = np.arange(reference_count)
    keep_count = max(3, int(np.ceil(reference_count * 0.7)))
    for _ in range(3):
        scale, rotation, translation = estimate_sim3(pred_centers[indices], gt_centers[indices])
        aligned_centers = scale * (pred_centers @ rotation.T) + translation
        residuals = np.linalg.norm(aligned_centers - gt_centers, axis=1)
        indices = np.argsort(residuals)[:keep_count]
    scale, rotation, translation = estimate_sim3(pred_centers[indices], gt_centers[indices])
    return apply_sim3_to_poses(pred_c2w, scale, rotation, translation)


def camera_dragging_metrics(
    pred_c2w: np.ndarray, gt_c2w: np.ndarray, reference_count: int = 15
) -> Dict[str, np.ndarray]:
    """Measure signed movement of each camera toward the opposite GT cluster."""
    pred_aligned = align_poses_on_reference(pred_c2w, gt_c2w, reference_count)
    pred_centers = pred_aligned[:, :3, 3]
    gt_centers = np.asarray(gt_c2w, dtype=np.float64)[:, :3, 3]
    reference_center = gt_centers[:reference_count].mean(axis=0)
    doppelganger_center = gt_centers[reference_count:].mean(axis=0)
    separation_vector = doppelganger_center - reference_center
    separation = float(np.linalg.norm(separation_vector))
    if separation <= 1e-12:
        raise ValueError("GT reference and doppelganger camera centroids coincide")
    axis = separation_vector / separation
    midpoint_projection = float(np.dot((reference_center + doppelganger_center) * 0.5, axis))
    gt_projection = gt_centers @ axis
    pred_projection = pred_centers @ axis

    attraction = np.empty(len(gt_centers), dtype=np.float64)
    attraction[:reference_count] = (
        pred_projection[:reference_count] - gt_projection[:reference_count]
    ) / separation
    attraction[reference_count:] = (
        gt_projection[reference_count:] - pred_projection[reference_count:]
    ) / separation
    side_crossing = np.zeros(len(gt_centers), dtype=bool)
    side_crossing[:reference_count] = pred_projection[:reference_count] > midpoint_projection
    side_crossing[reference_count:] = pred_projection[reference_count:] < midpoint_projection
    normalized_center_error = np.linalg.norm(pred_centers - gt_centers, axis=1) / separation
    predicted_separation = float(
        np.dot(
            pred_centers[reference_count:].mean(axis=0) - pred_centers[:reference_count].mean(axis=0),
            axis,
        )
    )
    return {
        "aligned_c2w": pred_aligned,
        "axis": axis,
        "gt_cluster_separation": separation,
        "attraction_fraction": attraction,
        "normalized_center_error": normalized_center_error,
        "side_crossing": side_crossing,
        "group_separation_retention": predicted_separation / separation,
    }
