"""Small geometry helpers for paper-ready point-cloud exports."""

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import trimesh


def estimate_sim3(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate dst = scale * rotation @ src + translation with Umeyama."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("src and dst must be matching Nx3 arrays with N >= 3")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = src_centered.T @ dst_centered / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(rotation) < 0:
        sign[-1] = -1.0
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if variance < 1e-12:
        raise ValueError("Cannot estimate Sim(3) from coincident camera centers")
    scale = float(np.dot(singular_values, sign) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_sim3_to_points(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return scale * (points @ rotation.T) + translation


def apply_sim3_to_poses(
    poses_c2w: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    poses = np.asarray(poses_c2w, dtype=np.float64)
    aligned = poses.copy()
    aligned[:, :3, :3] = rotation @ poses[:, :3, :3]
    aligned[:, :3, 3] = apply_sim3_to_points(poses[:, :3, 3], scale, rotation, translation)
    return aligned


def camera_display_scale(poses_c2w: np.ndarray) -> float:
    centers = np.asarray(poses_c2w, dtype=np.float64)[:, :3, 3]
    diagonal = float(np.linalg.norm(np.ptp(centers, axis=0)))
    return max(0.04 * diagonal, 1e-3)


def _segment_mesh(start: np.ndarray, end: np.ndarray, radius: float, color) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, segment=np.stack([start, end]), sections=6)
    mesh.visual.face_colors = np.asarray(color, dtype=np.uint8)
    return mesh


def camera_frusta_mesh(
    poses_c2w: np.ndarray,
    size: float,
    color: Iterable[int],
    aspect: float = 4.0 / 3.0,
) -> trimesh.Trimesh:
    """Create solid camera frusta for OpenCV-style +Z camera poses."""
    depth = float(size)
    half_height = 0.42 * depth
    half_width = half_height * aspect
    local = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
        ],
        dtype=np.float64,
    )
    edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    radius = max(depth * 0.018, 1e-5)
    meshes = []
    for pose in np.asarray(poses_c2w, dtype=np.float64):
        world = local @ pose[:3, :3].T + pose[:3, 3]
        meshes.extend(_segment_mesh(world[a], world[b], radius, color) for a, b in edges)
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes)


def camera_error_lines_mesh(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    indices: np.ndarray,
    radius: float,
) -> trimesh.Trimesh:
    """Connect predicted and GT camera centers for selected frames."""
    meshes = []
    for index in np.asarray(indices, dtype=np.int64):
        start = pred_c2w[index, :3, 3]
        end = gt_c2w[index, :3, 3]
        if np.linalg.norm(end - start) <= 1e-10:
            continue
        meshes.append(_segment_mesh(start, end, radius, (255, 220, 0, 255)))
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes)


def export_visualization_geometry(
    output_dir: Path,
    points: np.ndarray,
    colors: np.ndarray,
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    reference_indices: np.ndarray,
    doppelganger_indices: np.ndarray,
    point_frame_indices: np.ndarray,
    image_names,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    cloud = trimesh.points.PointCloud(points, colors=colors)
    cloud.export(output_dir / "pointcloud.ply")

    point_frame_indices = np.asarray(point_frame_indices, dtype=np.int64)
    reference_mask = np.isin(point_frame_indices, reference_indices)
    reference_cloud = trimesh.points.PointCloud(points[reference_mask], colors=colors[reference_mask])
    reference_cloud.export(output_dir / "reference_pointcloud.ply")

    palette = np.asarray(
        [
            [255, 70, 55, 255],
            [255, 170, 0, 255],
            [220, 60, 210, 255],
            [80, 220, 90, 255],
            [150, 90, 255, 255],
        ],
        dtype=np.uint8,
    )
    doppel_clouds = []
    doppel_points = []
    doppel_colors = []
    for palette_index, frame_index in enumerate(doppelganger_indices):
        mask = point_frame_indices == frame_index
        solid_color = np.repeat(palette[palette_index % len(palette)][None], int(mask.sum()), axis=0)
        frame_cloud = trimesh.points.PointCloud(points[mask], colors=solid_color)
        doppel_clouds.append((int(frame_index), frame_cloud))
        doppel_points.append(points[mask])
        doppel_colors.append(solid_color)
    if doppel_points:
        trimesh.points.PointCloud(
            np.concatenate(doppel_points, axis=0), colors=np.concatenate(doppel_colors, axis=0)
        ).export(output_dir / "doppelganger_pointcloud.ply")

    size = camera_display_scale(gt_c2w)
    pred_reference = camera_frusta_mesh(pred_c2w[reference_indices], size, (0, 170, 255, 255))
    pred_doppelganger = camera_frusta_mesh(pred_c2w[doppelganger_indices], size, (255, 70, 55, 255))
    gt = camera_frusta_mesh(gt_c2w, size * 0.82, (185, 185, 185, 255))
    error_lines = camera_error_lines_mesh(pred_c2w, gt_c2w, doppelganger_indices, size * 0.012)
    predicted = trimesh.util.concatenate([pred_reference, pred_doppelganger])
    predicted.export(output_dir / "predicted_cameras.ply")
    gt.export(output_dir / "gt_cameras.ply")

    scene = trimesh.Scene()
    scene.add_geometry(reference_cloud, geom_name="reference_pointcloud_rgb", node_name="reference_pointcloud_rgb")
    for frame_index, frame_cloud in doppel_clouds:
        stem = Path(image_names[frame_index]).stem
        name = f"doppelganger_points_{frame_index:02d}_{stem}"
        scene.add_geometry(frame_cloud, geom_name=name, node_name=name)
    scene.add_geometry(pred_reference, geom_name="predicted_reference_cameras", node_name="predicted_reference_cameras")
    scene.add_geometry(pred_doppelganger, geom_name="predicted_doppelganger_cameras", node_name="predicted_doppelganger_cameras")
    scene.add_geometry(gt, geom_name="colmap_gt_cameras", node_name="colmap_gt_cameras")
    if not error_lines.is_empty:
        scene.add_geometry(error_lines, geom_name="doppelganger_camera_error_lines", node_name="doppelganger_camera_error_lines")
    scene.export(output_dir / "scene.glb")
