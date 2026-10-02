import hydra
import os
import os.path as osp
import argparse
import sys
import torch
import logging
import json
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_videodepth
from utils.files import get_all_sequences, list_imgs_a_sequence
from utils.messages import set_default_arg
from videodepth.utils import save_depth_maps

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
    if pretrained_model_name_or_path.endswith('.pth'):
        ckpt = torch.load(pretrained_model_name_or_path)
        model.load_state_dict(ckpt, strict=False)
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
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/videodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = args.pretrained_weight or hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/videodepth.yaml
    output_dir: str                    = args.save_dir or hydra_cfg.output_dir
    device: str                        = args.device or hydra_cfg.device
    hydra_cfg.device = device

    # 0. create model
    model = load_model(pretrained_model_name_or_path).to(device).eval()
    logger = logging.getLogger("videodepth-infer")
    logger.info(f"Loaded model from {pretrained_model_name_or_path}")
    logger.info(f"Using device: {device}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        if dataset_info.type == "video":
            # most of the datasets have many sequences of video
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            raise ValueError("dataset type `mono` is not supported for videodepth evaluation")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        model = model.eval()
        output_root = osp.join(output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering videodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        # 3. infer for each sequence (video)
        for seq_idx, seq in enumerate(seq_list, start=1):
            filelist = list_imgs_a_sequence(dataset_info, seq)
            save_dir = osp.join(output_root, seq)

            if not hydra_cfg.overwrite and (osp.isdir(save_dir) and len(os.listdir(save_dir)) == 2 * len(filelist) + 1):
                logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} already processed, skipping.")
                continue
            
            # time_used: float, or List[float] (len = 2)
            # depth_maps: (N, H, W), torch.Tensor
            # conf_self: (N, H, W) torch.Tensor, or just None is ok
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            time_used, depth_maps, conf_self = infer_videodepth(filelist, model, hydra_cfg)
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if str(device).startswith("cuda") else 0.0
            logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} processed, time: {time_used}, peak memory: {peak_mem_mb} MB, saving depth maps...")

            os.makedirs(save_dir, exist_ok=True)
            save_depth_maps(depth_maps, save_dir, conf_self=conf_self)
            # save time
            with open(osp.join(save_dir, "_time.json"), "w") as f:
                json.dump({
                    "time": time_used,
                    "inference-time": time_used,
                    "frames": len(filelist),
                    "peak_mem_mb": peak_mem_mb,
                    "pretrained_model_name_or_path": pretrained_model_name_or_path,
                }, f, indent=4)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    LOCAL_ARGS = parse_local_args()
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()