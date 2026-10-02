import os
import os.path as osp
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple, Union

import imageio.v2
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile
from torch.utils.data import Dataset

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from datasets.utils.cropping import resize_image_depth_and_intrinsic
from utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float32,
    )


def read_tum_groundtruth(path: str) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = []
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            if len(values) != 8:
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = values
            cam_to_world = np.eye(4, dtype=np.float32)
            cam_to_world[:3, :3] = quat_xyzw_to_matrix(np.array([qx, qy, qz, qw], dtype=np.float32))
            cam_to_world[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
            timestamps.append(ts)
            poses.append(cam_to_world)
    order = np.argsort(timestamps)
    return np.asarray(timestamps, dtype=np.float64)[order], np.asarray(poses, dtype=np.float32)[order]


def nearest_pose(timestamp: float, gt_timestamps: np.ndarray, gt_poses: np.ndarray) -> np.ndarray:
    idx = bisect_left(gt_timestamps, timestamp)
    if idx <= 0:
        return gt_poses[0]
    if idx >= len(gt_timestamps):
        return gt_poses[-1]
    before = idx - 1
    after = idx
    if abs(gt_timestamps[before] - timestamp) <= abs(gt_timestamps[after] - timestamp):
        return gt_poses[before]
    return gt_poses[after]


class ETH3DSLAMRGBD(Dataset):
    def __init__(
        self,
        ETH3D_SLAM_DIR: str,
        sequences: Optional[List[str]] = None,
        load_img_size: int = 518,
        cache_file: str = "data/dataset_cache/eth3d_slam_rgbd_cache.npy",
    ):
        self.ETH3D_SLAM_DIR = ETH3D_SLAM_DIR
        if ETH3D_SLAM_DIR is None:
            raise NotImplementedError
        print(f"ETH3D_SLAM_DIR is {ETH3D_SLAM_DIR}")

        if osp.exists(cache_file):
            print(f"[ETH3D-SLAM-RGBD] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
        else:
            print(f"[ETH3D-SLAM-RGBD] Cache file not found, loading from {ETH3D_SLAM_DIR}")
            self.metadata = self._build_metadata(ETH3D_SLAM_DIR)
            os.makedirs(osp.dirname(cache_file), exist_ok=True)
            np.save(cache_file, self.metadata)

        if sequences is not None:
            sequence_set = set(sequences)
            self.metadata = {k: v for k, v in self.metadata.items() if k in sequence_set}
        self.sequence_list = sorted(self.metadata.keys())
        self.load_img_size = load_img_size
        print(f"[ETH3D-SLAM-RGBD] Data size: {len(self)}")

    @staticmethod
    def _build_metadata(root_dir: str) -> Dict[str, dict]:
        metadata = {}
        root = osp.abspath(root_dir)
        candidates = [root]
        candidates.extend(
            osp.join(root, item)
            for item in os.listdir(root)
            if osp.isdir(osp.join(root, item))
        )
        for seq_dir in candidates:
            if not (
                osp.isdir(osp.join(seq_dir, "rgb"))
                and osp.isdir(osp.join(seq_dir, "depth"))
                and osp.exists(osp.join(seq_dir, "calibration.txt"))
                and osp.exists(osp.join(seq_dir, "groundtruth.txt"))
            ):
                continue
            seq_name = osp.basename(seq_dir.rstrip(os.sep))
            with open(osp.join(seq_dir, "calibration.txt"), "r") as f:
                fx, fy, cx, cy = [float(x) for x in f.readline().split()[:4]]
            rgb_rows = []
            rgb_txt = osp.join(seq_dir, "rgb.txt")
            with open(rgb_txt, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    ts, rel_path = line.split()[:2]
                    basename = osp.basename(rel_path)
                    if osp.exists(osp.join(seq_dir, "rgb", basename)) and osp.exists(osp.join(seq_dir, "depth", basename)):
                        rgb_rows.append((float(ts), basename))
            gt_timestamps, gt_cam_to_world = read_tum_groundtruth(osp.join(seq_dir, "groundtruth.txt"))
            metadata[seq_name] = {
                "seq_dir": seq_dir,
                "frames": rgb_rows,
                "intrinsic": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32),
                "gt_timestamps": gt_timestamps,
                "gt_cam_to_world": gt_cam_to_world,
            }
        return metadata

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return len(self.metadata[sequence_name]["frames"])

    def __getitem__(self, idx_N):
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        ids = np.random.choice(self.get_seq_framenum(sequence_name=sequence_name), n_per_seq, replace=False)
        return self.get_data(sequence_name=sequence_name, ids=ids)

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Union[List[int], np.ndarray, None] = None,
    ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]

        meta = self.metadata[sequence_name]
        frames = meta["frames"]
        seq_len = len(frames)
        if ids is None:
            ids = np.arange(seq_len).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()

        image_paths = [""] * len(ids)
        images = [0] * len(ids)
        depths = [0] * len(ids)
        extrinsics = np.zeros((len(ids), 3, 4), dtype=np.float32)
        intrinsics = np.zeros((len(ids), 3, 3), dtype=np.float32)

        for id_index, frame_id in enumerate(ids):
            timestamp, img_name = frames[frame_id]
            impath = osp.join(meta["seq_dir"], "rgb", img_name)
            depthpath = osp.join(meta["seq_dir"], "depth", img_name)

            rgb_image = Image.open(impath)
            depthmap = imageio.v2.imread(depthpath)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 5000.0
            depthmap[depthmap < 1e-4] = 0

            intrinsic = meta["intrinsic"].copy()
            rgb_image, depthmap, intrinsic = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsic,
                output_width=self.load_img_size,
            )

            cam_to_world = nearest_pose(timestamp, meta["gt_timestamps"], meta["gt_cam_to_world"])
            world_to_cam = closed_form_inverse_se3(cam_to_world[None])[0]

            image_paths[id_index] = impath
            images[id_index] = to_tensor(rgb_image)
            depths[id_index] = depthmap
            intrinsics[id_index] = intrinsic
            extrinsics[id_index] = world_to_cam[:3, :]

        depths = np.array(depths)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics,
        )

        batch = {"seq_id": sequence_name, "seq_len": seq_len, "ind": torch.tensor(ids)}
        batch["image_paths"] = image_paths
        batch["images"] = torch.stack(images, dim=0)
        batch["pointclouds"] = pointclouds
        batch["valid_mask"] = depths > 1e-4
        return batch
