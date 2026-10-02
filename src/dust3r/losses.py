from copy import copy, deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import (
    inv,
    geotrf,
    normalize_pointcloud,
    normalize_pointcloud_group,
)
from dust3r.utils.geometry import (
    get_group_pointcloud_depth,
    get_group_pointcloud_center_scale,
    weighted_procrustes,
)
import numpy as np
import lpips
from dust3r.utils.camera import (
    pose_encoding_to_camera,
    camera_to_pose_encoding,
    relative_pose_absT_quatR,
)
import logging

from typing import *
import math

from dust3r.utils.pi3geometry import homogenize_points, se3_inverse, depth_edge
from dust3r.utils.alignment import align_points_scale

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.

    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                [-hard_max, hard_max] will be clamped. If None, 
                                no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor

    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    """L-norm loss"""

    def forward(self, a, b):
        assert (
            a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3
        ), f"Bad shape = {a.shape}"
        dist = self.distance(a, b)
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss(LLoss):
    """Euclidean distance between 3d points"""

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class MSELoss(LLoss):
    def distance(self, a, b):
        return (a - b) ** 2


MSE = MSELoss()


class Criterion(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(
            criterion, BaseCriterion
        ), f"{criterion} is not a proper criterion!"
        self.criterion = copy(criterion)

    def get_name(self):
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode="none"):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class RGBLoss(Criterion, MultiLoss):
    def __init__(self, criterion):
        super().__init__(criterion)
        self.ssim = SSIM()

    def img_loss(self, a, b):
        return self.criterion(a, b)

    def compute_loss(self, gts, preds, **kw):
        gt_rgbs = [gt["img"].permute(0, 2, 3, 1) for gt in gts]
        pred_rgbs = [pred["rgb"] for pred in preds]
        ls = [
            self.img_loss(pred_rgb, gt_rgb)
            for pred_rgb, gt_rgb in zip(pred_rgbs, gt_rgbs)
        ]
        details = {}
        self_name = type(self).__name__
        for i, l in enumerate(ls):
            details[self_name + f"_rgb/{i+1}"] = float(l)
            details[f"pred_rgb_{i+1}"] = pred_rgbs[i]
        rgb_loss = sum(ls) / len(ls)
        return rgb_loss, details


class DepthScaleShiftInvLoss(BaseCriterion):
    """scale and shift invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 3, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def normalize(self, x, mask):
        x_valid = x[mask]
        splits = mask.sum(dim=(1, 2)).tolist()
        x_valid_list = torch.split(x_valid, splits)
        shift = [x.mean() for x in x_valid_list]
        x_valid_centered = [x - m for x, m in zip(x_valid_list, shift)]
        scale = [x.abs().mean() for x in x_valid_centered]
        scale = torch.stack(scale)
        shift = torch.stack(shift)
        x = (x - shift.view(-1, 1, 1)) / scale.view(-1, 1, 1).clamp(min=1e-6)
        return x

    def distance(self, pred, gt, mask):
        pred = self.normalize(pred, mask)
        gt = self.normalize(gt, mask)
        return torch.abs((pred - gt)[mask])


class ScaleInvLoss(BaseCriterion):
    """scale invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 4, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, pred, gt, mask):
        pred_norm_factor = (torch.norm(pred, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        gt_norm_factor = (torch.norm(gt, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        pred = pred / pred_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        gt = gt / gt_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        return torch.norm(pred - gt, dim=-1)[mask]


class Regr3DPose(Criterion, MultiLoss):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        sky_loss_value=2,
        max_metric_scale=False,
    ):
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.gt_scale = gt_scale

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_norm_factor_point_cloud(
        self, pts_cross, valids, conf_cross, norm_self_only=False
    ):
        pts = [x for x in pts_cross]
        valids = [x for x in valids]
        confs = [x for x in conf_cross]
        norm_factor = normalize_pointcloud_group(
            pts, self.norm_mode, valids, confs, ret_factor_only=True
        )
        return norm_factor

    def get_norm_factor_poses(self, gt_trans, pr_trans, not_metric_mask):

        if self.norm_mode and not self.gt_scale:
            gt_trans = [x[:, None, None, :].clone() for x in gt_trans]
            valids = [torch.ones_like(x[..., 0], dtype=torch.bool) for x in gt_trans]
            norm_factor_gt = (
                normalize_pointcloud_group(
                    gt_trans,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
        else:
            norm_factor_gt = torch.ones(
                len(gt_trans), dtype=gt_trans[0].dtype, device=gt_trans[0].device
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            pr_trans_not_metric = [
                x[not_metric_mask][:, None, None, :].clone() for x in pr_trans
            ]
            valids = [
                torch.ones_like(x[..., 0], dtype=torch.bool)
                for x in pr_trans_not_metric
            ]
            norm_factor_pr_not_metric = (
                normalize_pointcloud_group(
                    pr_trans_not_metric,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric
        return norm_factor_gt, norm_factor_pr

    def get_all_pts3d(
        self,
        gts,
        preds,
        dist_clip=None,
        norm_self_only=False,
        norm_pose_separately=False,
        eps=1e-3,
        camera1=None,
    ):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"]) if camera1 is None else inv(camera1)
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        # valids = torch.stack(valids, dim=0)  # S B H W
        # valids = valids.permute(1, 0, 2, 3)  # B S H W
        # valids_masks = preprocess_mask(valids, mode="pad") # (B, S, H, W)
        #
        # valids = torch.unbind(valids_masks, dim=1) # [S] (B, H, W)

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_cross,
                valids,
                conf_cross,
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            norm_factor_pr_not_metric = self.get_norm_factor_point_cloud(
                [pr_pt_cross[not_metric_mask] for pr_pt_cross in pr_pts_cross],
                [valid[not_metric_mask] for valid in valids],
                [conf[not_metric_mask] for conf in conf_cross],
                norm_self_only=norm_self_only,
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, not_metric_mask
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(gt_trans, pr_trans, not_metric_mask)
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )

        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]
        pose_masks = (pose_norm_factor_gt.squeeze(-1) > eps) & (
            pose_norm_factor_pr.squeeze(-1) > eps
        )


        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_cross,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {},
        )

    def get_all_pts3d_with_scale_loss(
        self,
        gts,
        preds,
        dist_clip=None,
        norm_self_only=False,
        norm_pose_separately=False,
        eps=1e-3,
    ):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"])
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self[:1],
                gt_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        if self.norm_mode:
            norm_factor_pr = self.get_norm_factor_point_cloud(
                pr_pts_self[:1],
                pr_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            raise NotImplementedError
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pts_scale_loss = torch.abs(
                norm_factor_pr[~not_metric_mask] - norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pts_scale_loss = 0.0

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, torch.ones_like(not_metric_mask)
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(
                    gt_trans, pr_trans, torch.ones_like(not_metric_mask)
                )
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pose_scale_loss = torch.abs(
                pose_norm_factor_pr[~not_metric_mask]
                - pose_norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pose_scale_loss = 0.0
        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]

        pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
            pose_norm_factor_pr.squeeze() > eps
        )

        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),
                    gt,
                )
                for gt in gt_pts_self
            ]
            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {"scale_loss": pose_scale_loss + pts_scale_loss},
        )

    def compute_relative_pose_loss(
        self, gt_trans, gt_quats, pr_trans, pr_quats, masks=None
    ):
        if masks is None:
            masks = torch.ones(len(gt_trans), dtype=torch.bool, device=gt_trans.device)
        gt_trans_matrix1 = gt_trans[:, :, None, :].repeat(1, 1, gt_trans.shape[1], 1)[
            masks
        ]
        gt_trans_matrix2 = gt_trans[:, None, :, :].repeat(1, gt_trans.shape[1], 1, 1)[
            masks
        ]
        gt_quats_matrix1 = gt_quats[:, :, None, :].repeat(1, 1, gt_quats.shape[1], 1)[
            masks
        ]
        gt_quats_matrix2 = gt_quats[:, None, :, :].repeat(1, gt_quats.shape[1], 1, 1)[
            masks
        ]
        pr_trans_matrix1 = pr_trans[:, :, None, :].repeat(1, 1, pr_trans.shape[1], 1)[
            masks
        ]
        pr_trans_matrix2 = pr_trans[:, None, :, :].repeat(1, pr_trans.shape[1], 1, 1)[
            masks
        ]
        pr_quats_matrix1 = pr_quats[:, :, None, :].repeat(1, 1, pr_quats.shape[1], 1)[
            masks
        ]
        pr_quats_matrix2 = pr_quats[:, None, :, :].repeat(1, pr_quats.shape[1], 1, 1)[
            masks
        ]

        gt_rel_trans, gt_rel_quats = relative_pose_absT_quatR(
            gt_trans_matrix1, gt_quats_matrix1, gt_trans_matrix2, gt_quats_matrix2
        )
        pr_rel_trans, pr_rel_quats = relative_pose_absT_quatR(
            pr_trans_matrix1, pr_quats_matrix1, pr_trans_matrix2, pr_quats_matrix2
        )
        rel_trans_err = torch.norm(gt_rel_trans - pr_rel_trans, dim=-1)
        rel_quats_err = torch.norm(gt_rel_quats - pr_rel_quats, dim=-1)
        return rel_trans_err.mean() + rel_quats_err.mean()

    def compute_pose_loss(self, gt_poses, pred_poses, masks=None):
        """
        gt_pose: list of (Bx3, Bx4)
        pred_pose: list of (Bx3, Bx4)
        masks: None, or B
        """
        gt_trans = torch.stack([gt[0] for gt in gt_poses], dim=1)  # BxNx3
        gt_quats = torch.stack([gt[1] for gt in gt_poses], dim=1)  # BXNX3
        pred_trans = torch.stack([pr[0] for pr in pred_poses], dim=1)  # BxNx4
        pred_quats = torch.stack([pr[1] for pr in pred_poses], dim=1)  # BxNx4
        if masks == None:
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1).mean()
                + torch.norm(pred_quats - gt_quats, dim=-1).mean()
            )
        else:
            if not any(masks):
                return torch.tensor(0.0)
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1)[masks].mean()
                + torch.norm(pred_quats - gt_quats, dim=-1)[masks].mean()
            )

        return pose_loss

    def compute_loss(self, gts, preds, **kw):
        (
            gt_pts_cross,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks = [mask | sky for mask, sky in zip(masks, skys)]


        # if self.sky_loss_value > 0:
        #     assert (
        #         self.criterion.reduction == "none"
        #     ), "sky_loss_value should be 0 if no conf loss"
        #     for i, l in enumerate(ls_self):
        #         ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        self_name = type(self).__name__

        details = {}

        # cross view loss and details
        camera_only = gts[0]["camera_only"]
        pred_pts_cross = [pred_pts[~camera_only] for pred_pts in pred_pts_cross]
        gt_pts_cross = [gt_pts[~camera_only] for gt_pts in gt_pts_cross]
        masks_cross = [mask[~camera_only] for mask in masks]
        skys_cross = [sky[~camera_only] for sky in skys]

        if "Quantile" in self.criterion.__class__.__name__:
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            ls_cross, _ = self.criterion(
                pred_pts_cross, gt_pts_cross, masks_cross, None
            )
        else:
            ls_cross = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(
                    pred_pts_cross, gt_pts_cross, masks_cross
                )
            ]

        for i in range(len(ls_cross)):
            details[f"gt_img{i + 1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"valid_mask_{i + 1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i + 1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i + 1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i + 1}"] = preds[i]["desc"].detach()

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        ls = ls_cross
        masks = masks_cross
        details["img_ids"] = (
            np.arange(len(ls_cross)).tolist()
        )
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        return Sum(*list(zip(ls, masks))), (details | monitoring)


class Regr3DPoseBatchList(Regr3DPose):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        sky_loss_value=2,
        max_metric_scale=False,
    ):
        super().__init__(
            criterion, norm_mode, gt_scale, sky_loss_value, max_metric_scale
        )
        self.depth_only_criterion = DepthScaleShiftInvLoss()
        self.single_view_criterion = ScaleInvLoss()

    def reorg(self, ls_b, masks_b):
        ids_split = [mask.sum(dim=(1, 2)) for mask in masks_b]
        ls = [[] for _ in range(len(masks_b[0]))]
        for i in range(len(ls_b)):
            ls_splitted_i = torch.split(ls_b[i], ids_split[i].tolist())
            for j in range(len(masks_b[0])):
                ls[j].append(ls_splitted_i[j])
        ls = [torch.cat(l) for l in ls]
        return ls

    def compute_loss(self, gts, preds, **kw):
        (
            gt_pts_cross,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        camera_only = gts[0]["camera_only"]
        depth_only = gts[0]["depth_only"]
        single_view = gts[0]["single_view"]
        is_metric = gts[0]["is_metric"]

        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            raise NotImplementedError
        else:
            # list [(B, h, w, 3)] x num_views -> list [num_views, h, w, 3] x B
            masks_b = torch.unbind(torch.stack(masks, dim=1), dim=0)


        self_name = type(self).__name__

        gt_pts_cross_b = torch.unbind(
            torch.stack(gt_pts_cross, dim=1)[~camera_only], dim=0
        )
        pred_pts_cross_b = torch.unbind(
            torch.stack(pred_pts_cross, dim=1)[~camera_only], dim=0
        )
        masks_cross_b = torch.unbind(torch.stack(masks, dim=1)[~camera_only], dim=0)
        ls_cross_b = []
        for i in range(len(gt_pts_cross_b)):
            if depth_only[~camera_only][i]:
                ls_cross_b.append(
                    self.depth_only_criterion(
                        pred_pts_cross_b[i][..., -1],
                        gt_pts_cross_b[i][..., -1],
                        masks_cross_b[i],
                    )
                )
            elif single_view[~camera_only][i] and not is_metric[~camera_only][i]:
                ls_cross_b.append(
                    self.single_view_criterion(
                        pred_pts_cross_b[i], gt_pts_cross_b[i], masks_cross_b[i]
                    )
                )
            else:
                ls_cross_b.append(
                    self.criterion(
                        pred_pts_cross_b[i][masks_cross_b[i]],
                        gt_pts_cross_b[i][masks_cross_b[i]],
                    )
                )
        ls_cross = self.reorg(ls_cross_b, masks_cross_b)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks_cross = [mask[~camera_only] for mask in masks]
            skys_cross = [sky[~camera_only] for sky in skys]
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        details = {}
        for i in range(len(ls_cross)):
            details[f"gt_img{i + 1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"valid_mask_{i + 1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i + 1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i + 1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i + 1}"] = preds[i]["desc"].detach()

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        ls = ls_cross
        masks = masks_cross
        details["img_ids"] = (
            np.arange(len(ls_cross)).tolist()
        )
        pose_masks = pose_masks * gts[i]["img_mask"]
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        return Sum(*list(zip(ls, masks))), (details | monitoring)


class ConfLoss(MultiLoss):
    """Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10)

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction("none")

    def get_name(self):
        return f"ConfLoss({self.pixel_loss})"

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gts, preds, **kw):
        # compute per-pixel loss
        losses_and_masks, details = self.pixel_loss(gts, preds, **kw)
        if "is_self" in details and "img_ids" in details:
            img_ids = details["img_ids"]
        else:
            img_ids = list(range(len(losses_and_masks)))

        # weight by confidence
        conf_losses = []

        for i in range(len(losses_and_masks)):
            pred = preds[img_ids[i]]
            conf_key = "conf"

            camera_only = gts[0]["camera_only"]
            conf, log_conf = self.get_conf_log(
                pred[conf_key][~camera_only][losses_and_masks[i][1]]
            )

            conf_loss = losses_and_masks[i][0] * conf - self.alpha * log_conf
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            conf_losses.append(conf_loss)



            details[self.get_name() + f"_conf_loss/{img_ids[i]+1}"] = float(
                conf_loss
            )

        details.pop("img_ids", None)

        final_loss = sum(conf_losses) / len(conf_losses) * 2.0
        if "pose_loss" in details:
            final_loss = (
                final_loss + details["pose_loss"].clip(max=0.3) * 5.0
            )  # , details
        if "scale_loss" in details:
            final_loss = final_loss + details["scale_loss"]
        return final_loss, details


class Regr3DPose_ScaleInv(Regr3DPose):
    """Same than Regr3D but invariant to depth shift.
    if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gts, preds):
        # compute depth-normalized points
        (
            gt_pts_cross,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = super().get_all_pts3d(gts, preds)

        # measure scene scale

        _, gt_scale_cross = get_group_pointcloud_center_scale(gt_pts_cross, masks)
        _, pred_scale_cross = get_group_pointcloud_center_scale(pr_pts_cross, masks)

        # prevent predictions to be in a ridiculous range
        pred_scale_cross = pred_scale_cross.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:

            pr_pts_cross = [
                pr_pt_cross * gt_scale_cross / pred_scale_cross
                for pr_pt_cross in pr_pts_cross
            ]
        else:
            gt_pts_cross = [
                gt_pt_cross / gt_scale_cross for gt_pt_cross in gt_pts_cross
            ]
            pr_pts_cross = [
                pr_pt_cross / pred_scale_cross for pr_pt_cross in pr_pts_cross
            ]

        return (
            gt_pts_cross,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        )
    
def closed_form_scale_and_shift(pred, gt):
    """ 
    Args:
        pred:   (B, H, W, C) 
        gt:     (B, H, W, C) 
        valid_mask: (B, H, W) 
    Returns:
        scale:  (B,) 
        shift:  (B,) 
    """
    assert pred.dim() == 4 and gt.dim() == 4, "Inputs must be 4D tensors"
    B, H, W, C = pred.shape
    device = pred.device

    pred_flat = pred.view(-1, C)  # (N, C)
    gt_flat = gt.view(-1, C)  # (N, C)

    if C == 1: 
        pred_mean = pred_flat.mean(dim=0)
        gt_mean = gt_flat.mean(dim=0)

        numerator = ((pred_flat - pred_mean) * (gt_flat - gt_mean)).sum(dim=0)
        denominator = ((pred_flat - pred_mean) ** 2).sum(dim=0).clamp(min=1e-6)
        scale = numerator / denominator

        shift = gt_mean - scale * pred_mean
        return scale, shift

    elif C == 3:
        pred_mean = pred_flat.mean(0)
        gt_mean = gt_flat.mean(0)
        pred_centered = pred_flat - pred_mean
        gt_centered = gt_flat - gt_mean

        scale = (pred_centered * gt_centered).sum() / (pred_centered ** 2).sum().clamp(min=1e-6)
        shift = gt_mean - scale * pred_mean
        return scale, shift

    else:
        raise ValueError(f"Unsupported channel dimension C={C}. Only 1 or 3 channels are supported.")

def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
    """
    pts3d: B, H, W, 3
    valid_mask: B, H, W
    """
    dist = pts3d.norm(dim=-1)
    dist_sum = (dist * valid_mask).sum(dim=[1,2])
    valid_count = valid_mask.sum(dim=[1,2])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    # avg_scale = avg_scale.view(-1, 1, 1, 1, 1)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1)
    return pts3d, avg_scale

def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.cuda.amp.autocast(enabled=False):
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        normals = F.normalize(normals, p=2, dim=-1, eps=eps)


        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1)

    return normals, valids

class CameraLoss(nn.Module):
    def __init__(self, delta=1e-1, weights=(1.0, 1.0, 0.5)):
        super().__init__()
        self.weights = weights
    def forward(self, pred_pose, gt_pose):
        loss_T = (pred_pose[..., :3] - gt_pose[..., :3]).abs()
        loss_R = (pred_pose[..., 3:7] - gt_pose[..., 3:7]).abs()
        loss_FL = (pred_pose[..., 7:] - gt_pose[..., 7:]).abs()

        loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
        loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
        loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

        # Clamp outlier translation loss to prevent instability, then average
        loss_T = loss_T.clamp(max=100).mean()
        loss_R = loss_R.mean()
        loss_FL = loss_FL.mean()
        return (self.weights[0] * loss_T + self.weights[1] * loss_R + self.weights[2] * loss_FL)

class DepthOrPmapLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.grad_scales = 3
        self.gamma = 1.0 

    def gradient_loss_multi_scale(self, pred, gt, mask=None):
        total = 0
        for s in range(self.grad_scales):
            step = 2 ** s
            pred_s = pred[:, ::step, ::step]
            gt_s = gt[:, ::step, ::step]
            mask_s = mask[:, ::step, ::step]
            total += self.normal_loss(pred_s, gt_s, mask_s)
        return total / self.grad_scales

    def normal_loss(self, pred, gt, mask=None):
        pred_norm, _ = point_map_to_normal(pred, mask)
        gt_norm, _ = point_map_to_normal(gt, mask)
        cos_sim = F.cosine_similarity(pred_norm, gt_norm, dim=-1)
        return 1 - cos_sim.mean()

    def image_gradient_loss(self, pred, gt, mask=None):
        assert pred.dim() == 4 and pred.shape[-1] == 1
        assert gt.shape == pred.shape

        B, H, W, _ = pred.shape
        device = pred.device

        dx_pred = pred[:, :, 1:] - pred[:, :, :-1]  # [B,H,W-1,1]
        dx_gt = gt[:, :, 1:] - gt[:, :, :-1]
        dx_mask = mask[:, :, 1:] & mask[:, :, :-1]  # [B,H,W-1]

        dy_pred = pred[:, 1:, :] - pred[:, :-1, :]  # [B,H-1,W,1]
        dy_gt = gt[:, 1:, :] - gt[:, :-1, :]
        dy_mask = mask[:, 1:, :] & mask[:, :-1, :]  # [B,H-1,W]

        min_h = min(dy_pred.shape[1], dx_pred.shape[1])
        min_w = min(dx_pred.shape[2], dy_pred.shape[2])

        dx_pred = dx_pred[:, :min_h, :min_w, :]  # [B,H-1,W-1,1]
        dx_gt = dx_gt[:, :min_h, :min_w, :]
        dx_mask = dx_mask[:, :min_h, :min_w]  # [B,H-1,W-1]

        dy_pred = dy_pred[:, :min_h, :min_w, :]  # [B,H-1,W-1,1]
        dy_gt = dy_gt[:, :min_h, :min_w, :]
        dy_mask = dy_mask[:, :min_h, :min_w]  # [B,H-1,W-1]

        loss_dx = F.l1_loss(dx_pred * dx_mask.unsqueeze(-1),
                            dx_gt * dx_mask.unsqueeze(-1))
        loss_dy = F.l1_loss(dy_pred * dy_mask.unsqueeze(-1),
                            dy_gt * dy_mask.unsqueeze(-1))

        return (loss_dx + loss_dy) / 2

    def forward(self, pred, gt, sigma_p=None, sigma_g=None, valid_mask=None):
        if self.training:
            pred_normalized, _ = normalize_pointcloud(pred, valid_mask)
            gt_normalized, _ = normalize_pointcloud(gt, valid_mask)
        else:
            pred_normalized, gt_normalized = pred, gt
        scale, shift = closed_form_scale_and_shift(
            pred_normalized, gt_normalized
        )
        pred_aligned = pred_normalized * scale + shift
        sigma_p = sigma_p.clamp(min=1e-6)
        if sigma_g is not None:
            sigma_g = sigma_g.clamp(min=1e-6)
        #sigma = 0.5 * (sigma_p + sigma_g)
        sigma = sigma_p
        diff = (pred_aligned - gt_normalized).abs()

        C = diff.shape[-1]

        main_loss = (sigma[..., None].expand(-1, -1, -1, C) * diff)[valid_mask[..., None].expand(-1, -1, -1, C)].mean()

        if pred.shape[-1] == 1:
            grad_loss = self.image_gradient_loss(pred_aligned, gt_normalized, valid_mask)
        else:
            grad_loss = self.gradient_loss_multi_scale(pred_aligned, gt_normalized, valid_mask)
        reg_loss = -self.alpha * torch.log(sigma.clamp(min=1e-6))[valid_mask].mean()
        # return main + reg
        return self.gamma * main_loss + grad_loss + reg_loss

class TrackLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = 0.2
        self.gamma = 1.0
    def forward(self, y_pr, y_gt, vis_pr, vis_gt, w_p, w_g):
        #w = 0.5 * (w_p + w_g)
        w = w_p
        l_pos = (y_pr - y_gt).norm(dim=-1)
        l_pos = (w * l_pos).mean()

        l_vis = self.bce(vis_pr, vis_gt.float())
        l_vis = (w * l_vis).mean()
        return l_pos + l_vis


# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# # ---------------------------------------------------------------------------
# # PointLoss: Scale-invariant Local Pointmap
# # ---------------------------------------------------------------------------

class Pi3PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        self.train_conf = train_conf
        if self.train_conf:
            # self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh
    # def prepare_segformer(self):
    #     from pi3.models.segformer.model import EncoderDecoder

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            
            final_loss += 0.05 * (local_conf_loss)
            details['local_conf_loss'] = (local_conf_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_loss = self.noraml_loss(aligned_local_pts, gt_local_pts, valid_masks)
        final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

# # ---------------------------------------------------------------------------
# # CameraLoss: Affine-invariant Camera Pose
# # ---------------------------------------------------------------------------

class Pi3CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)



class DistillPi3Loss(MultiLoss):
    def __init__(
        self,
        train_conf=True,
        feature_weight=0.0,
        feature_reg_weight=1.0,
        feature_img_weight=1.0,
        feature_patch_start_idx=5,
        point_weight=10.0,
        camera_weight=20.0,
    ):
        super().__init__()
        self.point_loss = Pi3PointLoss(train_conf=train_conf)
        self.camera_loss = Pi3CameraLoss()
        self.point_weight = point_weight
        self.camera_weight = camera_weight
        self.feature_weight = feature_weight
        self.feature_reg_weight = feature_reg_weight
        self.feature_img_weight = feature_img_weight
        self.feature_patch_start_idx = feature_patch_start_idx

    def get_name(self): return "DistillPi3Loss"

    def decoder_feature_loss(self, pred, gt):
        if self.feature_weight <= 0:
            return None, {}
        if 'decoder_features' not in pred or 'decoder_features' not in gt:
            return None, {}

        pred_feat = pred['decoder_features'].float()
        gt_feat = gt['decoder_features'].float()
        split_idx = min(self.feature_patch_start_idx, pred_feat.shape[1], gt_feat.shape[1])

        reg_pred, img_pred = pred_feat[:, :split_idx], pred_feat[:, split_idx:]
        reg_gt, img_gt = gt_feat[:, :split_idx], gt_feat[:, split_idx:]

        total = pred_feat.new_zeros(())
        details = {}

        if reg_pred.numel() > 0 and self.feature_reg_weight > 0:
            reg_loss = 1 - F.cosine_similarity(reg_pred, reg_gt, dim=-1).mean()
            total = total + self.feature_reg_weight * reg_loss
            details['Lfeature_reg'] = (
                float(reg_loss.detach()) * self.feature_reg_weight * self.feature_weight
            )

        if img_pred.numel() > 0 and self.feature_img_weight > 0:
            img_loss = 1 - F.cosine_similarity(img_pred, img_gt, dim=-1).mean()
            total = total + self.feature_img_weight * img_loss
            details['Lfeature_img'] = (
                float(img_loss.detach()) * self.feature_img_weight * self.feature_weight
            )

        with torch.no_grad():
            if reg_pred.numel() > 0:
                details.update({
                    'feat_reg_mse_raw': float(F.mse_loss(reg_pred, reg_gt)),
                    'feat_reg_mse_norm': float(F.mse_loss(F.normalize(reg_pred, dim=-1), F.normalize(reg_gt, dim=-1))),
                    'feat_reg_norm_pred': float(reg_pred.norm(dim=-1).mean()),
                    'feat_reg_norm_teacher': float(reg_gt.norm(dim=-1).mean()),
                    'feat_reg_cos': float(F.cosine_similarity(reg_pred, reg_gt, dim=-1).mean()),
                })
            if img_pred.numel() > 0:
                details.update({
                    'feat_img_mse_raw': float(F.mse_loss(img_pred, img_gt)),
                    'feat_img_mse_norm': float(F.mse_loss(F.normalize(img_pred, dim=-1), F.normalize(img_gt, dim=-1))),
                    'feat_img_norm_pred': float(img_pred.norm(dim=-1).mean()),
                    'feat_img_norm_teacher': float(img_gt.norm(dim=-1).mean()),
                    'feat_img_cos': float(F.cosine_similarity(img_pred, img_gt, dim=-1).mean()),
                })

        return total, details
    
    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        
        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
    

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
        )
    
    def normalize_pred(self, pred):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = pred['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred
    
    def compute_loss(self, gts, preds):
        # ---------- Lpmap ----------
        if 'global_points' in gts:
            gt = self.normalize_pred(gts)
        else:
            gt = self.prepare_gt(gts)
        pred = self.normalize_pred(preds)

        total = 0.0
        details = dict()

        #Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        total += point_loss * self.point_weight
        details.update(point_loss_details)
        details['Lpoint'] = float(point_loss.detach()) * self.point_weight

        # ---------- Lcamera ----------
        camera_loss, camera_loss_details = self.camera_loss(preds, gts, scale)
        total += camera_loss * self.camera_weight
        details.update(camera_loss_details)
        details['Lcamera'] = float(camera_loss.detach()) * self.camera_weight

        feature_loss, feature_loss_details = self.decoder_feature_loss(preds, gts)
        if feature_loss is not None:
            total += feature_loss * self.feature_weight
            details.update(feature_loss_details)
            details['Lfeature'] = float(feature_loss.detach()) * self.feature_weight
        # point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        # total += point_loss
        # details.update(point_loss_details)

        # # ---------- Lcamera ----------
        # camera_loss, camera_loss_details = self.camera_loss(preds, gts, scale)
        # total += camera_loss * 0.1
        # details.update(camera_loss_details)
       
        return total, details


class FinetunePi3Loss(DistillPi3Loss):
    def get_name(self): return "FinetunePi3Loss"

    def compute_loss(self, gts, preds):
        gt = self.prepare_gt(gts)
        pred = self.normalize_pred(preds)

        total = 0.0
        details = dict()

        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        total += point_loss * 10
        details.update(point_loss_details)

        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        total += camera_loss * 20
        details.update(camera_loss_details)

        details = {
            key: float(val.detach()) if torch.is_tensor(val) else val
            for key, val in details.items()
        }
        details["total"] = float(total.detach())

        return total, details
