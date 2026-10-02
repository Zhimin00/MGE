# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import warnings

from torch import Tensor
from torch import nn
import torch

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


def summarize_token_retention(original_tokens: int, retained_tokens: int, layer_idx=None) -> dict:
    """Build diagnostics from the actual sequence lengths around a merge."""
    original_tokens = int(original_tokens)
    retained_tokens = int(retained_tokens)
    if original_tokens <= 0:
        raise ValueError("original_tokens must be positive")

    merged_tokens = original_tokens - retained_tokens
    return {
        "layer": int(layer_idx) if layer_idx is not None else None,
        "original_tokens": original_tokens,
        "retained_tokens": retained_tokens,
        "merged_tokens": merged_tokens,
        "retained_ratio": retained_tokens / original_tokens,
        "actual_merge_ratio": merged_tokens / original_tokens,
    }


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = [qkv[:,:,i] for i in range(3)]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        q, k, v = [qkv[:,:,i] for i in range(3)]

        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.transpose(1, 3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttentionRope(AttentionRope):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
        merge_ratio: float = 0.0,
        patch_width: int = 37,
        patch_height: int = 28,
        merge_strategy: str = "aga",
        protected_frame_ratio: float = 0.1,
        keep_first_anchor: bool = True,
        anchor_selection: str = "uniform",
        protected_patch_ratio: float = 0.1,
        merge_stride: int = 2,
        outlier_weight: float = 1.0,
        edge_weight: float = 0.5,
    ) -> None:
        super().__init__(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=proj_drop, qk_norm=qk_norm,
            norm_layer=norm_layer, rope=rope,
        )
        self.merge_ratio = merge_ratio
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.merge_strategy = merge_strategy
        self.protected_frame_ratio = protected_frame_ratio
        self.keep_first_anchor = keep_first_anchor
        self.anchor_selection = anchor_selection
        self.anchor_indices = None
        self.protected_patch_ratio = protected_patch_ratio
        self.merge_stride = merge_stride
        self.outlier_weight = outlier_weight
        self.edge_weight = edge_weight

    def _pi3_geometry_merge_ratio(self, layer_idx) -> float:
        if layer_idx is None:
            return float(self.merge_ratio)
        if layer_idx <= 9:
            cap = 0.9
        elif layer_idx <= 25:
            cap = 0.8
        elif layer_idx <= 29:
            cap = 0.6
        else:
            cap = 0.3
        return min(float(self.merge_ratio), cap)

    def forward(self, x: Tensor, attn_bias=None, xpos=None, attn_mask=None, global_merging=None) -> Tensor:
        B, N, C = x.shape
        original_token_count = N
        self.last_merge_stats = None
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        # Token merging (same pattern as fastvggt)
        if global_merging is not None:
            # print(self.merge_strategy)
            from fastpi3.merging.merge import token_merge_bipartite2d

            generator = torch.Generator(device=x.device)
            generator.manual_seed(33)

            if self.merge_strategy in {"ablation_uniform"}:
                from fastpi3.merging.merge import token_merge_frame_protected_bipartite2d

                r = int(x.shape[1] * self.merge_ratio)
                m, u = token_merge_frame_protected_bipartite2d(
                    x,
                    self.patch_width,
                    self.patch_height,
                    2, 2, r, False, generator,
                    enable_frame_protection=True,
                    protected_frame_ratio=self.protected_frame_ratio,
                )
            elif self.merge_strategy in {"ablation_fixed"}:
                from fastpi3.merging.frame_merge import (
                    legacy_original_retained_token_count,
                    token_merge_frame_adaptive_global_bipartite2d,
                )

                r = int(x.shape[1] * self.merge_ratio)
                tokens_per_frame = self.patch_width * self.patch_height + 5
                num_frames = x.shape[1] // tokens_per_frame
                baseline_retained_tokens = legacy_original_retained_token_count(
                    num_frames,
                    self.patch_width,
                    self.patch_height,
                    self.merge_stride,
                    self.merge_stride,
                    self.merge_ratio,
                )
                m, u = token_merge_frame_adaptive_global_bipartite2d(
                    x,
                    self.patch_width,
                    self.patch_height,
                    self.merge_stride,
                    self.merge_stride,
                    r,
                    False,
                    generator,
                    anchor_frame_ratio=self.protected_frame_ratio,
                    anchor_merge_ratio=0.0,
                    non_anchor_merge_ratio=self.merge_ratio,
                    keep_first_anchor=self.keep_first_anchor,
                    anchor_selection=self.anchor_selection,
                    anchor_indices=self.anchor_indices,
                    adaptive_by_anchor_similarity=False,
                    target_total_merge_count=x.shape[1] - baseline_retained_tokens,
                )
            elif self.merge_strategy in {
                "aga",
                "ablation_adaptive",
            }:
                from fastpi3.merging.frame_merge import (
                    legacy_original_retained_token_count,
                    token_merge_frame_adaptive_global_bipartite2d,
                )

                r = int(x.shape[1] * self.merge_ratio)
                tokens_per_frame = self.patch_width * self.patch_height + 5
                num_frames = x.shape[1] // tokens_per_frame
                original_retained_tokens = legacy_original_retained_token_count(
                    num_frames,
                    self.patch_width,
                    self.patch_height,
                    self.merge_stride,
                    self.merge_stride,
                    self.merge_ratio,
                )
                m, u = token_merge_frame_adaptive_global_bipartite2d(
                    x,
                    self.patch_width,
                    self.patch_height,
                    self.merge_stride,
                    self.merge_stride,
                    r,
                    False,
                    generator,
                    anchor_frame_ratio=self.protected_frame_ratio,
                    anchor_merge_ratio=0.0,
                    non_anchor_merge_ratio=self.merge_ratio,
                    keep_first_anchor=self.keep_first_anchor,
                    anchor_selection=self.anchor_selection,
                    anchor_indices=self.anchor_indices,
                    adaptive_by_anchor_similarity=True,
                    min_non_anchor_merge_ratio=max(0.1, self.merge_ratio * 0.35),
                    max_non_anchor_merge_ratio=0.75,
                    target_total_merge_count=x.shape[1] - original_retained_tokens,
                )
            elif self.merge_strategy in {"original", "ablation_baseline"}:
                # print('original fast')
                r = int(x.shape[1] * self.merge_ratio)

                m, u = token_merge_bipartite2d(
                x,
                self.patch_width,
                self.patch_height,
                2, 2, r, False, generator,
                enable_protection=True,
            )
            else:
                raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

            m_a, u_a = (m, u)

            B_q, H_q, N_q, D_q = q.shape

            q_merge_in = q.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
            k_merge_in = k.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
            v_merge_in = v.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)

            q_out, k_out, v_out = m_a(
                q_merge_in,
                mode="mean",
                extra_tensors=k_merge_in,
                extra_tensors_2=v_merge_in,
            )

            del q_merge_in, k_merge_in, v_merge_in

            N_m = q_out.shape[1]
            q = q_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
            k = k_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
            v = v_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)

            del q_out, k_out, v_out

            self.last_merge_stats = summarize_token_retention(
                original_token_count,
                N_m,
                global_merging,
            )
            N = N_m

        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        del q, k, v
        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        if global_merging is not None:
            x = u_a(x)

        return x

    def _frame_sparse_sdpa(self, q, k, v, allowed_frame, num_frames):
        """Per-frame sparse SDPA. Each query frame attends only to allowed frames.

        q, k, v: (B, H, N_tokens, Dh)
        allowed_frame: (num_frames, num_frames) bool, True = allowed
        Returns: (B, H, N_tokens, Dh)
        """
        B, H, Ntokens, Dh = q.shape
        T = Ntokens // num_frames
        qF = q.view(B, H, num_frames, T, Dh)
        kF = k.view(B, H, num_frames, T, Dh)
        vF = v.view(B, H, num_frames, T, Dh)
        out = torch.empty_like(qF)

        if allowed_frame.dtype != torch.bool:
            allowed_frame = allowed_frame.bool()
        eye = torch.eye(num_frames, device=q.device, dtype=torch.bool)
        allowed_frame = allowed_frame | eye  # ensure self-attention

        for i in range(num_frames):
            js = torch.nonzero(allowed_frame[i], as_tuple=False).squeeze(-1)
            if js.numel() == 0:
                js = torch.tensor([i], device=q.device)

            kb = kF[:, :, js].reshape(B, H, js.numel() * T, Dh)
            vb = vF[:, :, js].reshape(B, H, js.numel() * T, Dh)
            qb = qF[:, :, i]

            if q.dtype == torch.bfloat16:
                with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    ob = scaled_dot_product_attention(qb, kb, vb)
            else:
                with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    ob = scaled_dot_product_attention(qb, kb, vb)
            out[:, :, i] = ob

        return out.reshape(B, H, Ntokens, Dh)


class CrossAttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        B, N, C = query.shape
        _, M, _ = key.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(query, key, value, attn_bias, qpos, kpos)

        B, N, C = query.shape
        _, M, _ = key.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_attn_score(blk_class, x, frame_num, token_length, xpos=None):
    x = blk_class.norm1(x)

    B, N, C = x.shape
    qkv = blk_class.attn.qkv(x).reshape(B, N, 3, blk_class.attn.num_heads, C // blk_class.attn.num_heads)

    qkv = qkv.transpose(1, 3)
    q, k, v = [qkv[:,:,i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    attn = (q.permute(0, 2, 1, 3) * blk_class.attn.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1))
    attn = attn.reshape(B, blk_class.attn.num_heads, frame_num, token_length, frame_num, token_length)
    return attn
