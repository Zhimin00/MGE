# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
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
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


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

        # q, k, v = unbind(qkv, 2)
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

        # q, k, v = unbind(qkv, 2)
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


"""
Following is written by GPT-4o
"""
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

        # Separate projection layers for query, key, and value
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
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, M)
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(query, key, value, attn_bias)

        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
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

        # Compute memory-efficient attention
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, C)

        # Final projection
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
        # q, k, v = unbind(qkv, 2)
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

        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(frame_num, 261, frame_num, 261).mean(dim=[1, 3]).sum(1)         # for frame attention matrix
        # global_valid_id = torch.where(score_matrix > 0)
        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class FlashAttentionRope(AttentionRope):
    """
    Self-attention with RoPE and optional frame-level masking.

    When `frame_attn_mask` is provided (bool Tensor, shape [num_frames, num_frames]),
    attention is computed only over allowed frame pairs.
    For each allowed pair (frame_i, frame_j), a hw x hw block attention is computed
    and results are accumulated.  A single final softmax normalises across all
    contributed key-frames.  This avoids materialising a full (num_frames * hw)^2
    attention matrix.

    When `frame_attn_mask` is None, falls back to a single SDPA call.
    """

    def forward(self, x: Tensor, attn_bias=None, xpos=None,
                attn_mask=None, past_key_values=None,
                use_cache=False) -> Tensor:
        B, Ntokens, C = x.shape

        qkv = self.qkv(x).reshape(B, Ntokens, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        pos_k = xpos
        if use_cache:
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)
            if past_key_values is not None:
                past_k, past_v = past_key_values
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            new_kv = (k, v)
            a, b, c, d, e = k.shape
            k = k.reshape(a, b, c*d, e)
            v = v.reshape(a, b, c*d, e)
            if pos_k is not None:
                pos_k = pos_k.repeat(1, c, 1)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        # --- Frame-level sparse attention ----------------------------------
        num_frames = None
        if attn_mask is not None and num_frames is not None:
            x = self._frame_sparse_sdpa(q, k, v, attn_mask,
                                          num_frames)

        # --- Full attention (original path) ---------------------------------
        else:
            if attn_mask is not None:
                with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                if q.dtype == torch.bfloat16:
                    with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        x = scaled_dot_product_attention(q, k, v)
                else:
                    with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                        x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, Ntokens, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
            return x, new_kv
        return x

    # ------------------------------------------------------------------
    # def _frame_sparse_attn(self, q, k, v, frame_mask, num_frames,
    #                        B, Ntokens, C):
    #     """Compute attention only over allowed frame pairs.

    #     Parameters
    #     ----------
    #     q, k, v       : [B, num_heads, Ntokens, head_dim]
    #     frame_mask    : bool [num_frames, num_frames], True = allowed
    #     num_frames    : int  – number of frames (S)
    #     B             : total batch size
    #     Ntokens       : total tokens (= S * hw)
    #     C             : hidden dim

    #     Returns
    #     -------
    #     out : [B, Ntokens, C]
    #     """
    #     hw = Ntokens // num_frames
    #     H = self.num_heads
    #     head_dim = C // H
    #     device = q.device
    #     dtype = v.dtype
    #     scale = head_dim ** -0.5

    #     # Reshape to per-frame tensors: [B, num_frames, hw, H*head_dim]
    #     q_f = q.transpose(1, 2).reshape(B, Ntokens, H * head_dim)
    #     q_f = q_f.reshape(B, num_frames, hw, H * head_dim)
    #     k_f = k.transpose(1, 2).reshape(B, Ntokens, H * head_dim)
    #     k_f = k_f.reshape(B, num_frames, hw, H * head_dim)
    #     v_f = v.transpose(1, 2).reshape(B, Ntokens, H * head_dim)
    #     v_f = v_f.reshape(B, num_frames, hw, H * head_dim)

    #     # Accumulators – one entry per token
    #     weighted_sum = torch.zeros(B, Ntokens, H * head_dim,
    #                               device=device, dtype=dtype)
    #     logsumexp_buf = torch.full((B, Ntokens, 1),
    #                               float("-inf"), device=device, dtype=torch.float32)

    #     # Iterate over allowed (query_frame, key_frame) pairs
    #     allowed = torch.where(frame_mask)
    #     for qi, ki in zip(allowed[0].cpu(), allowed[1].cpu()):
    #         qi, ki = qi.item(), ki.item()

    #         q_block = q_f[:, qi, :, :]            # [B, hw, D]
    #         k_block = k_f[:, ki, :, :]
    #         v_block = v_f[:, ki, :, :]

    #         # Scaled QK^T
    #         logits = torch.matmul(q_block * scale,
    #                              k_block.transpose(1, 2))  # [B, hw, hw]
    #         probs, lse = torch.softmax(logits, dim=-1,
    #                                    dtype=torch.float32,
    #                                    return_lse=True)     # [B,hw,hw], [B,hw]

    #         out_block = torch.matmul(probs, v_block)         # [B, hw, D]

    #         weighted_sum[:, qi * hw:(qi + 1) * hw, :] += out_block
    #         logsumexp_buf[:, qi * hw:(qi + 1) * hw, :] += lse.unsqueeze(2)

    #     # Final normalisation across all key-frames
    #     denom = torch.exp(logsumexp_buf)
    #     out = weighted_sum / denom.clamp(min=1e-5)          # [B, Ntokens, D]

    #     # Project back to [B, Ntokens, C]
    #     out = out.reshape(B, Ntokens, H, head_dim)
    #     out = out.transpose(1, 2).reshape(B, Ntokens, C)
    #     return self.proj_drop(self.proj(out))


    def _frame_sparse_sdpa(self, q, k, v, allowed_frame, num_frames):
        """
        q,k,v: (B, H, S, T, Dh)
        allowed_frame: (S, S) bool, True=allowed
        topk_frames: int or None. If set, cap attended frames per query frame.
        Returns: (B, H, S, T, Dh)
        """
        B, H, Ntokens, Dh = q.shape
        device = q.device
        
        T = Ntokens // num_frames
        qF = q.view(B, H, num_frames, T, Dh)
        kF = k.view(B, H, num_frames, T, Dh)
        vF = v.view(B, H, num_frames, T, Dh)
        out = torch.empty_like(qF)
        # Make sure diagonal is allowed (self-attend)
        if allowed_frame.dtype != torch.bool:
            allowed_frame = allowed_frame.bool()
        eye = torch.eye(num_frames, device=device, dtype=torch.bool)
        allowed_frame = allowed_frame | eye  # ensure self allowed
        for i in range(num_frames):
            js = torch.nonzero(allowed_frame[i], as_tuple=False).squeeze(-1)  # [M]
            if js.numel() == 0:
                js = torch.tensor([i], device=device)

            # Gather K,V tokens from the selected frames
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

def get_attn_score(blk_class, x, frame_num, token_length, xpos=None):
    x = blk_class.norm1(x)

    B, N, C = x.shape
    qkv = blk_class.attn.qkv(x).reshape(B, N, 3, blk_class.attn.num_heads, C // blk_class.attn.num_heads)

    qkv = qkv.transpose(1, 3)
    # q, k, v = unbind(qkv, 2)
    q, k, v = [qkv[:,:,i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    # score = (q.permute(0, 2, 1, 3) * blk_class.attn.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(B, frame_num, token_length, frame_num, token_length).mean(dim=[2, 4]).sum(-1)

    attn = (q.permute(0, 2, 1, 3) * blk_class.attn.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1))## B, Heads, N, N
    attn = attn.reshape(B, blk_class.attn.num_heads, frame_num, token_length, frame_num, token_length)
    return attn
