import os
import os.path as osp
import logging
import argparse
import sys
import random
import time
import numpy as np
import torch
import hydra

from tqdm import tqdm
from omegaconf import DictConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_cameras_w2c
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg, write_csv, save_list_of_matrices
from relpose.evo_utils import calculate_averages, load_traj, eval_metrics, plot_trajectory, get_tum_poses, save_tum_poses

LOCAL_ARGS = None


def parse_local_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained-weight", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args, hydra_argv = parser.parse_known_args()

    # Leave Hydra overrides in sys.argv, but remove local argparse-only flags.
    sys.argv = [sys.argv[0]] + hydra_argv
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    args = LOCAL_ARGS or argparse.Namespace(
        device=None, pretrained_weight=None, save_dir=None
    )
    set_seed(hydra_cfg.seed)

    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose-distance.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    device = args.device if args.device else hydra_cfg.device
    hydra_cfg.device = device
    pretrained_model_name_or_path: str = args.pretrained_weight or hydra_cfg.pi3.pretrained_model_name_or_path
    output_dir = args.save_dir if args.save_dir else hydra_cfg.output_dir

    # 0. create model
    # model = Pi3.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()
    
    # model = VGGT.from_pretrained("facebook/VGGT-1B")
    if pretrained_model_name_or_path.endswith('.pth'):
        ckpt = torch.load(pretrained_model_name_or_path)
        model.load_state_dict(ckpt, strict=False)
    elif pretrained_model_name_or_path.endswith('.safetensors'):
        from safetensors.torch import load_model
        model = Pi3()
        # This loads the weights directly into your model instance
        load_model(model, pretrained_model_name_or_path)
    else:
        model = Pi3.from_pretrained(pretrained_model_name_or_path)
    model = model.to(device).eval()
    logger = logging.getLogger(f"relpose-dist")
    logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")
    logger.info(f"Using device: {device}")
    logger.info(f"Using output dir: {output_dir}")
    logger.info(f"Using random seed: {hydra_cfg.seed}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        seq_list = get_all_sequences(dataset_info)
        output_root = osp.join(output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)
        seq_metric_file = osp.join(output_root, "seq_metrics.csv")
        if osp.exists(seq_metric_file):
            os.remove(seq_metric_file)

        # 3. infer for each sequence
        model = model.eval()
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering relpose(c2w) on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        results = []
        all_data_dict = {
            "ATE": 0.0,
            "RPE trans": 0.0,
            "RPE rot": 0.0,
            "inference-time": 0.0,
            "peak-mem-mb": 0.0,
        }
        tbar = tqdm(seq_list, desc=f"[{dataset_name} eval]")
        for seq in tbar:
            # 4.1 list all images of this sequence
            filelist = list_imgs_a_sequence(dataset_info, seq)
            filelist = filelist[:: hydra_cfg.pose_eval_stride]

            # 4.2 real inference
            # pr_poses: c2w poses, (N, 3, 4), in torch
            # pr_intrs: focals + pps, (N, 3, 3), in numpy
            is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.time()
            pr_poses, pr_intrs = infer_cameras_w2c(filelist, model, hydra_cfg) #infer_cameras_c2w
            if is_cuda:
                torch.cuda.synchronize()
            end = time.time()
            inference_time_ms = (end - start) * 1000
            peak_mem_mb = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if is_cuda
                else 0.0
            )
            pred_traj = get_tum_poses(pr_poses)

            # 4.3 save predicted poses & intrinsics
            seq_save_dir = osp.join(output_root, seq)
            os.makedirs(seq_save_dir, exist_ok=True)
            # save predicted poses
            save_tum_poses(pred_traj, osp.join(output_root, seq, "pred_traj.txt"), verbose=hydra_cfg.verbose)
            np.save(osp.join(seq_save_dir, "pred_poses.npy"), pr_poses)
            save_list_of_matrices(pr_poses.numpy().tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))
            # save predicted intrinsics (if available)
            if pr_intrs is not None:
                np.save(osp.join(seq_save_dir, "pred_intrinsics.npy"), pr_intrs)
                save_list_of_matrices(pr_intrs.tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))

            # 4.4 read ground truth trajectory
            try:
                gt_traj = load_traj(
                    gt_traj_file = dataset_info.anno.path.format(seq=seq),
                    traj_format  = dataset_info.anno.format,
                    stride       = hydra_cfg.pose_eval_stride,
                )
            except np.linalg.LinAlgError:
                logger.warning(f"Failed to load ground truth trajectory for sequence {seq} in dataset {dataset_name}.")
                continue

            # 4.5 evaluate predicted trajectory with ground truth trajectory, plot the trajectory
            if gt_traj is not None:
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj, gt_traj,
                    seq      = seq,
                    filename = osp.join(output_root, seq, "eval_metric.txt"),
                    verbose  = hydra_cfg.verbose,
                )
                plot_trajectory(pred_traj, gt_traj, title=seq, filename=osp.join(output_root, seq, "vis.png"), verbose=hydra_cfg.verbose)
            else:
                raise ValueError(f"Ground truth trajectory not found for sequence {seq} in dataset {dataset_name}.")

            # 4.6 save sequence metrics to csv
            seq_metrics = {
                "dataset": dataset_name,
                "seq": seq,
                "pretrained_model_name_or_path": pretrained_model_name_or_path,
                "ATE": ate,
                "RPE trans": rpe_trans,
                "RPE rot": rpe_rot,
                "inference-time": inference_time_ms,
                "peak-mem-mb": peak_mem_mb,
            }
            write_csv(seq_metric_file, seq_metrics)
            results.append((seq, ate, rpe_trans, rpe_rot))
            all_data_dict["ATE"] += ate
            all_data_dict["RPE trans"] += rpe_trans
            all_data_dict["RPE rot"] += rpe_rot
            all_data_dict["inference-time"] += inference_time_ms
            all_data_dict["peak-mem-mb"] += peak_mem_mb

            # 4.7. update metric for a sequence to tqdm bar
            tbar.set_postfix_str(f"Seq {seq} ATE: {ate:5.2f} | RPE-trans: {rpe_trans:5.2f} | RPE-rot: {rpe_rot:5.2f}")

        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
        num_samples = max(1, len(results))

        dataset_metrics = {
            "ATE": avg_ate,
            "RPE trans": avg_rpe_trans,
            "RPE rot": avg_rpe_rot,
            "inference-time": all_data_dict["inference-time"] / num_samples,
            "peak-mem-mb": all_data_dict["peak-mem-mb"] / num_samples,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
        }
        statistics_file = osp.join(output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, dataset_metrics)
        logger.info(f"{dataset_name} - Average pose estimation metrics: {dataset_metrics}")
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    LOCAL_ARGS = parse_local_args()
    set_default_arg("evaluation", "relpose-distance")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    with torch.no_grad():
        main()
