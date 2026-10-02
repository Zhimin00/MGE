import hydra
import os
import os.path as osp
import argparse
import sys
import time
import json
import numpy as np
import cv2
import logging
import torch

from tqdm import tqdm
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_monodepth
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg

LOCAL_ARGS = None


def parse_local_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained-weight", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args, hydra_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_argv
    return args


def load_model(pretrained_model_name_or_path: str):
    if pretrained_model_name_or_path.endswith(".pth"):
        ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in ckpt["model"].items()
        }
        model = Pi3()
        model.load_state_dict(state_dict, strict=True)
        return model
    if pretrained_model_name_or_path.endswith(".safetensors"):
        from safetensors.torch import load_model as load_safetensors_model

        model = Pi3()
        load_safetensors_model(model, pretrained_model_name_or_path)
        return model
    return Pi3.from_pretrained(pretrained_model_name_or_path)


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    args = LOCAL_ARGS or argparse.Namespace(device=None, pretrained_weight=None, save_dir=None)
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = args.pretrained_weight or hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/monodepth.yaml
    output_dir: str                    = args.save_dir or hydra_cfg.output_dir
    device: str                        = args.device or hydra_cfg.device
    hydra_cfg.device = device

    # 0. create model
    model = load_model(pretrained_model_name_or_path).to(device).eval()
    logger = logging.getLogger("monodepth-infer")
    logger.info(f"Loaded model from {pretrained_model_name_or_path}")
    logger.info(f"Using device: {device}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        if dataset_info.type == "video":
            # most of the datasets have many sequences of video
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            # some datasets (like nyu-v2) have only a set of images, only for monodepth
            seq_list = [None]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        # 3. infer for each sequence
        output_root = osp.join(output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering monodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")
        for seq_idx, seq in enumerate(seq_list):
            # 3.1 list the images in the sequence
            filelist = list_imgs_a_sequence(dataset_info, seq)
            save_dir = osp.join(output_root, seq) if seq is not None else output_root
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Processing {len(filelist)} images to {osp.relpath(save_dir, hydra_cfg.work_dir)}...")
            total_inference_time = 0.0
            peak_mem_mb = 0.0
            processed_frames = 0

            # 3.2 infer for each image
            for file in tqdm(filelist):
                # 3.2.1 skip if the file already exists
                npy_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.npy'))
                png_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.png'))
                if not hydra_cfg.overwrite and (osp.exists(npy_save_path) and osp.exists(png_save_path)):
                    continue

                # 3.2.2 infer the depth map
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.time()
                depth_map = infer_monodepth(file, model, hydra_cfg)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                total_inference_time += time.time() - start
                peak_mem_mb = max(
                    peak_mem_mb,
                    torch.cuda.max_memory_allocated() / 1024 / 1024 if str(device).startswith("cuda") else 0.0,
                )
                processed_frames += 1

                # 3.2.3 save the depth map to the save_dir as npy
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
                elif not isinstance(depth_map, np.ndarray):
                    raise ValueError(f"Unknown depth map type: {type(depth_map)}")
                np.save(npy_save_path, depth_map)

                # 3.2.4 also save the png
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_map = (depth_map * 255).astype(np.uint8)
                cv2.imwrite(png_save_path, depth_map)
            time_save_path = osp.join(save_dir, "_time.json")
            if processed_frames > 0 or not osp.exists(time_save_path):
                with open(time_save_path, "w") as f:
                    json.dump({
                        "time": total_inference_time,
                        "inference-time": total_inference_time,
                        "frames": processed_frames,
                        "total_frames": len(filelist),
                        "peak_mem_mb": peak_mem_mb,
                        "pretrained_model_name_or_path": pretrained_model_name_or_path,
                    }, f, indent=4)
        # for each dataset
        logger.info(f"Monodepth inference for dataset {dataset_name} finished!")

    del model
    torch.cuda.empty_cache()
    logger.info(f"Monodepth inference for Pi3 finished!")

if __name__ == "__main__":
    LOCAL_ARGS = parse_local_args()
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()
