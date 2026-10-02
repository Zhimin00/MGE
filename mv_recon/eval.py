import os
import json
import argparse
import sys
import random
import torch
import numpy as np
import open3d as o3d
import os.path as osp
import hydra
import logging

from omegaconf import DictConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from sparsepi3.models.sparsepi3 import SparsePi3
from utils.interfaces import infer_mv_pointclouds, infer_mv_pointclouds_vggt
from mv_recon.utils import umeyama, accuracy, completion
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto
LOCAL_ARGS = None


def parse_local_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sparse-ratio", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained-weight", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--inference-only", action="store_true")
    args, hydra_argv = parser.parse_known_args()
    # if not 0.0 <= args.attn_ratio <= 1.0:
    #     raise ValueError(f"--attn-ratio must be in [0, 1], got {args.attn_ratio}")

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
        device=None,
        pretrained_weight=None,
        save_dir=None,
        inference_only=False,
    )
    set_seed(hydra_cfg.seed)

    # CLI overrides for device, pretrained weight, and save directory
    device = args.device if args.device else hydra_cfg.device
    pretrained_model_name_or_path: str = args.pretrained_weight or hydra_cfg.pi3.pretrained_model_name_or_path
    output_dir = args.save_dir if args.save_dir else hydra_cfg.output_dir

    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data

    # 0. create model
    if pretrained_model_name_or_path.endswith('.pth'):
        ckpt = torch.load(pretrained_model_name_or_path)
        new_state_dict = dict()
        for k, v in ckpt['model'].items():
             name = k[7:] if k.startswith('module.') else k
             new_state_dict[name] = v
        model = Pi3()
        model.load_state_dict(new_state_dict, strict=True)
    elif pretrained_model_name_or_path.endswith('.safetensors'):
        # model = Pi3.from_pretrained("yyfz233/Pi3")
        from safetensors.torch import load_model
        model = Pi3()
        load_model(model, pretrained_model_name_or_path)

    model = model.to(device).eval()

    logger = logging.getLogger("mv_recon-eval")
    logger.info(f"Loaded model from {pretrained_model_name_or_path}")
    logger.info(f"Using device: {device}")
    logger.info(f"Using random seed: {hydra_cfg.seed}")
    logger.info(f"Inference-only mode: {args.inference_only}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 1.2 ready for output directory & metrics
        output_root = osp.join(output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)
        samples_file = osp.join(
            output_root,
            "_inference_samples.csv" if args.inference_only else "_all_samples.csv",
        )
        all_data_dict = {
            "inference-time": 0.0,
            "peak-mem-mb": 0.0,
        } if args.inference_only else {
            "Acc-mean":  0.0,  "Acc-med":  0.0,
            "Comp-mean": 0.0,  "Comp-med": 0.0,
            "NC-mean":   0.0,  "NC-med":   0.0,
            "NC1-mean":  0.0,  "NC1-med":  0.0,
            "NC2-mean":  0.0,  "NC2-med":  0.0,
            "inference-time": 0.0,
            "peak-mem-mb": 0.0,
        }

        # 1.3 load pre-sampled seq-id-map
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating Multi-View Pointcloud Reconstruction of Pi3 on dataset {dataset_name}...")
        sample_config: DictConfig = dataset_info.sampling
        logger.info(f"Sampling strategy: {sample_config.strategy}")
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map: dict = json.load(f)
        if osp.exists(samples_file):
            os.remove(samples_file)
        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            # 2. load data, choose specific ids of a sequence
            data = dataset.get_data(sequence_name=seq_name, ids=ids)
            filelist: list         = data['image_paths']  # [str] * N
            images: torch.Tensor   = data['images']       # (N, 3, H, W)
            gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
            valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

            # 3. real inference, predicted pointcloud aligned to ground truth (data_h, data_w)
            data_h, data_w         = images.shape[-2:]
            # import pdb; pdb.set_trace()
            pred_pts, inference_time_ms, peak_mem_mb = infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))#, attn_ratio=args.attn_ratio)  # (N, H, W, 3)
            if args.inference_only:
                seq_output_name = seq_name.replace("/", "-")
                write_csv(samples_file, {
                    "seq": seq_output_name,
                    "pretrained_model_name_or_path": pretrained_model_name_or_path,
                    "num-input-frames": len(ids),
                    "inference-time": inference_time_ms,
                    "peak-mem-mb": peak_mem_mb,
                })
                all_data_dict["inference-time"] += inference_time_ms
                all_data_dict["peak-mem-mb"] += peak_mem_mb
                logger.info(
                    f"[{dataset_name} {seq_idx}/{len(seq_id_map)}] Seq: {seq_output_name}, "
                    f"inference time: {inference_time_ms} ms, peak memory: {peak_mem_mb} MB"
                )
                del pred_pts
                torch.cuda.empty_cache()
                continue

            assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."
            if 'ETH' not in dataset_name:
                # print('crop')
                cx = data_h // 2
                cy = data_w // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                images = images[:, :, t:b, l:r]
                valid_mask = valid_mask[:, t:b, l:r]
                pred_pts = pred_pts[:, t:b, l:r, :]
                gt_pts = gt_pts[:, t:b, l:r, :]

            # 4. save input images
            seq_name = seq_name.replace("/", "-")
            save_image_grid_auto(images, osp.join(output_root, f"{seq_name}.png"))

            num_input_frames = len(ids)
            
            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

            # 5. coarse align
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

            # 6. filter invalid points
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)

            # 7. save predicted & ground truth point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-pred.ply"), pcd)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt.ply"), pcd_gt)

            # 8. ICP align refinement
            if "DTU" in dataset_name:
                threshold = 100
            else:
                threshold = 0.1

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)
            
            # 9. estimate normals
            pcd.estimate_normals()
            pcd_gt.estimate_normals()
            pred_normal = np.asarray(pcd.normals)
            gt_normal = np.asarray(pcd_gt.normals)

            # o3d.io.write_point_cloud(
            #     os.path.join(
            #         save_path, f"{seq.replace('/', '_')}-mask-icp.ply"
            #     ),
            #     pcd,
            # )

            # 10. compute metrics
            acc, acc_med, nc1, nc1_med = accuracy(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            comp, comp_med, nc2, nc2_med = completion(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            logger.info(
                f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
            )

            # 11. save metrics to csv
            write_csv(samples_file, {
                "seq":        seq_name,
                "pretrained_model_name_or_path": pretrained_model_name_or_path,
                "num-input-frames": num_input_frames,
                "Acc-mean":  acc,
                "Acc-med":   acc_med,
                "Comp-mean": comp,
                "Comp-med":  comp_med,
                "NC1-mean":  nc1,
                "NC1-med":   nc1_med,
                "NC2-mean":  nc2,
                "NC2-med":   nc2_med,
                "inference-time": inference_time_ms,
                "peak-mem-mb": peak_mem_mb,
            })
            all_data_dict["Acc-mean"]  += acc
            all_data_dict["Acc-med"]   += acc_med
            all_data_dict["Comp-mean"] += comp
            all_data_dict["Comp-med"]  += comp_med
            all_data_dict["NC-mean"]   += (nc1 + nc2) / 2
            all_data_dict["NC-med"]    += (nc1_med + nc2_med) / 2
            all_data_dict["NC1-mean"]  += nc1
            all_data_dict["NC1-med"]   += nc1_med
            all_data_dict["NC2-mean"]  += nc2
            all_data_dict["NC2-med"]   += nc2_med
            all_data_dict['inference-time'] += inference_time_ms
            all_data_dict['peak-mem-mb'] += peak_mem_mb

            # release cuda memory
            torch.cuda.empty_cache()

        num_samples = len(seq_id_map)
        metric_dict = {
            metric: value / num_samples
            for metric, value in all_data_dict.items()
            if metric != "model"
        }
        metric_dict["pretrained_model_name_or_path"] = pretrained_model_name_or_path

        result_kind = "inference" if args.inference_only else "metric"
        statistics_file = osp.join(output_dir, f"{dataset_name}-{result_kind}")
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)
    
    del model
    torch.cuda.empty_cache()
    logger.info(f"Finished evaluating Pi3 on all datasets.")


if __name__ == "__main__":
    LOCAL_ARGS = parse_local_args()
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
