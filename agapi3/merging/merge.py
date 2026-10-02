import torch
from typing import Tuple, Callable, Optional, Union
import math

@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]  # [B, chunk_size, C]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def do_nothing(
    x: torch.Tensor,
    mode: str = "mean",
    extra_tensors=None,
    extra_tensors_2=None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    elif extra_tensors is not None:
        return x, extra_tensors
    else:
        return x


def _expand_idx(idx_1d, B, c):
    """Expand 1D long index to [B, K, c] for torch.gather(dim=1) on [B, N, c]."""
    return idx_1d.view(1, -1, 1).expand(B, -1, c)


def within_frame_tome(
    sample_x: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    patch_h: int,
    patch_w: int,
    num_register: int,
    merge_ratio: float,
    stride: int = 2,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Callable, Callable]:
    """ToMe merge plan for multi-frame tokens, constrained within frames."""
    N_total = sample_x.shape[1]
    C = sample_x.shape[2]
    hw = patch_h * patch_w
    device = sample_x.device
    r_per_frame = max(0, int(hw * merge_ratio))

    if r_per_frame <= 0 or hw <= stride * stride:
        return do_nothing, do_nothing

    # Grid-based dst selection within each frame
    num_grid_h = max(1, patch_h // stride)
    num_grid_w = max(1, patch_w // stride)
    rand_h = torch.randint(0, stride, (num_grid_h * num_grid_w,),
       device=device, generator=generator)
    rand_w = torch.randint(0, stride, (num_grid_h * num_grid_w,),
       device=device, generator=generator)
    gh = torch.arange(num_grid_h, device=device)
    gw = torch.arange(num_grid_w, device=device)
    gh, gw = torch.meshgrid(gh, gw, indexing="ij")
    patch_dst_local = ((gh * stride + rand_h.reshape_as(gh)) * patch_w +
       (gw * stride + rand_w.reshape_as(gw))).flatten()

    # Per-frame: registers (always dst) + grid survivors = dst; rest = src
    is_dst = torch.ones(tokens_per_frame, dtype=torch.bool, device=device)
    is_patch_dst = torch.zeros(hw, dtype=torch.bool, device=device)
    is_patch_dst[patch_dst_local] = True
    is_dst[num_register:] = is_patch_dst

    src_local = torch.nonzero(~is_dst, as_tuple=False).flatten()
    dst_local = torch.nonzero(is_dst, as_tuple=False).flatten()

    # Expand to all frames
    frame_offsets = torch.arange(num_frames, device=device) * tokens_per_frame
    dst_global = (dst_local.unsqueeze(0) + frame_offsets.unsqueeze(1)).flatten()
    src_global = (src_local.unsqueeze(0) + frame_offsets.unsqueeze(1)).flatten()

    num_dst = dst_global.shape[0]
    num_src = src_global.shape[0]
    r_total = min(num_src, num_frames * r_per_frame)

    if num_src == 0 or r_total <= 0:
        return do_nothing, do_nothing

    # Cosine similarity: for each src, find most similar dst
    idx_src = _expand_idx(src_global, 1, C)
    idx_dst = _expand_idx(dst_global, 1, C)
    sample_src = torch.gather(sample_x, dim=1, index=idx_src)
    sample_dst = torch.gather(sample_x, dim=1, index=idx_dst)
    sample_src_n = sample_src / sample_src.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sample_dst_n = sample_dst / sample_dst.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Chunked max-similarity via JIT-compiled helper
    chunk_size = min(4000, num_src)
    dst_transposed = sample_dst_n.transpose(-1, -2)
    node_max, node_idx = fast_similarity_chunks(sample_src_n, dst_transposed, chunk_size)

    # Most redundant first
    edge_order = node_max.argsort(dim=-1, descending=True)[..., None]
    to_merge_e = edge_order[..., :r_total, :]
    to_keep_e = edge_order[..., r_total:, :]

    # merge_dst_e[k] = which dst absorbs the k-th merged src
    merge_dst_e = torch.gather(node_idx, dim=1,
       index=to_merge_e.squeeze(-1))  # [1, r_total]
    merge_dst_e = merge_dst_e.squeeze(0)  # [r_total]

    # merge_src_global[k] = global token index of k-th src to merge
    merge_src_global = torch.gather(src_global.unsqueeze(0), dim=1,
       index=to_merge_e.squeeze(-1)).squeeze(0)  # [r_total]

    # keep_src_global[k] = global token index of k-th src to retain
    keep_src_global = torch.gather(src_global.unsqueeze(0), dim=1,
       index=to_keep_e.squeeze(-1)).squeeze(0)  # [keep_cnt]

    keep_cnt = num_src - r_total

    # Position map: merged output index -> original token index
    merged_pos_map = torch.cat([keep_src_global, dst_global], dim=0).unsqueeze(0)  # [1, new_N]

    def merge_fn(t):
        B = t.shape[0]
        c = t.shape[2]
        dst_tokens = torch.gather(t, dim=1, index=_expand_idx(dst_global, B, c))
        src_to_merge = torch.gather(t, dim=1, index=_expand_idx(merge_src_global, B, c))
        src_to_keep = torch.gather(t, dim=1, index=_expand_idx(keep_src_global, B, c))
        dst_tokens = dst_tokens.scatter_reduce(
            -2, _expand_idx(merge_dst_e, B, c), src_to_merge, reduce="mean")
        return torch.cat([src_to_keep, dst_tokens], dim=1)

    def unmerge_fn(t):
        B = t.shape[0]
        c = t.shape[2]
        src_to_keep = t[..., :keep_cnt, :]
        dst_tokens = t[..., keep_cnt:, :]
        src_merged = torch.gather(dst_tokens, dim=-2,
           index=_expand_idx(merge_dst_e, B, c))
        out = torch.zeros(B, N_total, c, device=t.device, dtype=t.dtype)
        out.scatter_(dim=-2, index=_expand_idx(dst_global, B, c), src=dst_tokens)
        out.scatter_(dim=-2, index=_expand_idx(keep_src_global, B, c), src=src_to_keep)
        out.scatter_(dim=-2, index=_expand_idx(merge_src_global, B, c), src=src_merged)
        return out

    merge_fn.pos_map = merged_pos_map
    return merge_fn, unmerge_fn


def token_merge_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Divide tokens into source (src) and destination (dst) groups, and merge r tokens from src to dst.
    dst tokens are selected by randomly choosing one token from each (sx, sy) region.
    Optionally protect the top 10% of tokens from merging based on importance scores.

    Args:
        metric [B, N, C]: Tensor for similarity computation, B=batch size, N=token count, C=feature dimension
        w: Image width in tokens
        h: Image height in tokens
        sx: dst stride in x dimension, must divide w evenly
        sy: dst stride in y dimension, must divide h evenly
        r: Number of tokens to remove through merging
        no_rand: If True, disable randomness (use only top-left token)
        generator: Random number generator if no_rand is False and not None
        enable_protection: If True, enable importance protection feature

    Returns:
        (merge, unmerge): Two functions for merging tokens and restoring pre-merge state
    """
    B, N, _ = metric.shape  # Batch size B, total tokens N
    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather

    tokens_per_img = w * h + 5
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N, "Token count doesn't match (w*h+5)*num_imgs"

    with torch.no_grad():
        # Determine whether to compute importance scores based on enable_protection
        if enable_protection:
            num_protected = int(N * 0.1)
            step = max(1, N // num_protected)
            protected_indices = torch.arange(0, N, step, device=metric.device)[
                :num_protected
            ]
        else:
            protected_indices = None
            num_protected = 0

        # Global idx_buffer_seq of length N; -1 indicates dst, 0 indicates src (maintain original logic)
        idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx  # Number of blocks within each image

        # Mark first image entirely as dst
        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

        # Process other images - fully vectorized batch operations
        if num_imgs > 1:
            cls_indices = (
                torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
            )
            cls_indices = cls_indices[:, None] + torch.arange(5, device=metric.device)
            idx_buffer_seq[cls_indices.flatten()] = -1
            effective_h = min(hsy * sy, h)
            effective_w = min(wsx * sx, w)
            effective_grid_size = effective_h * effective_w

            if no_rand:
                base_pattern = torch.zeros(
                    effective_grid_size, device=metric.device, dtype=torch.int64
                )
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img + 5
                )
                grid_indices = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=metric.device
                )
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                    num_imgs - 1
                )
            else:
                total_other_imgs = num_imgs - 1
                all_rand_idx = torch.randint(
                    sy * sx,
                    size=(total_other_imgs, hsy, wsx),
                    device=metric.device,
                    generator=generator,
                )

                scatter_src = -torch.ones(
                    total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64
                )

                idx_buffer_batch = torch.zeros(
                    total_other_imgs,
                    hsy,
                    wsx,
                    sy * sx,
                    device=metric.device,
                    dtype=torch.int64,
                )
                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )

                idx_buffer_batch = (
                    idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(total_other_imgs, hsy * sy, wsx * sx)
                )

                # Batch fill to target positions - still needs a small loop here, but operations are greatly reduced
                for i in range(total_other_imgs):
                    img_idx = i + 1
                    grid_start = img_idx * tokens_per_img + 5
                    flat_view = idx_buffer_batch[
                        i, :effective_h, :effective_w
                    ].flatten()
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = (
                        flat_view
                    )

        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst_orig = int((idx_buffer_seq == -1).sum())

        # Original src and dst indices
        a_idx_orig = rand_idx[:, num_dst_orig:, :]
        b_idx_orig = rand_idx[:, :num_dst_orig, :]
        a_idx = a_idx_orig
        b_idx = b_idx_orig

        if enable_protection:
            protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        # Define an internal function to separate src, dst, and protected tokens
        def split(x):
            C = x.shape[-1]

            if enable_protection:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                protected = gather(
                    x, dim=1, index=protected_idx.expand(B, num_protected_actual, C)
                )
                return src, dst, protected
            else:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

        # Compute cosine similarity (normalize first then dot product)
        metric = metric / metric.norm(dim=-1, keepdim=True)
        if enable_protection:
            a, b, protected = split(metric)
        else:
            a, b = split(metric)

        r = min(a.shape[1], r)
        num_src_actual = a.shape[1]

        if num_src_actual == 0 or r <= 0:
            return do_nothing, do_nothing

        chunk_size = min(5000, num_src_actual)

        node_max = torch.empty(B, num_src_actual, device=a.device, dtype=a.dtype)
        node_idx = torch.empty(B, num_src_actual, device=a.device, dtype=torch.long)

        b_transposed = b.transpose(-1, -2)
        node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # If protection is enabled, filter out protected tokens to ensure they are not merged
        if enable_protection:
            src_indices = a_idx[0, :, 0]
            protected_mask_src = torch.isin(src_indices, protected_indices)
            edge_flat = edge_idx[0, :, 0]
            valid_mask = ~protected_mask_src[edge_flat]
            valid_edges = edge_flat[valid_mask]

            valid_count = valid_edges.shape[0]
            r_actual = min(r, valid_count)

            unm_idx = valid_edges[r_actual:].unsqueeze(0).unsqueeze(-1)
            src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
        else:
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            r_actual = r

        # Get dst token indices corresponding to each src token to be merged
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        r = r_actual

    # Define merge function to merge selected src tokens to corresponding dst tokens
    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if enable_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)

        n, t1, c = src.shape

        # Extract unmerged src tokens - using actual unm_idx size
        unm_len = unm_idx.shape[1]
        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        src_len = src_idx.shape[1]
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)

        # ---------------- Extra tensor processing ----------------
        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            E_dim = extra_tensors.shape[-1]
            if enable_protection:
                src_e, dst_e, protected_e = split(extra_tensors)
            else:
                src_e, dst_e = split(extra_tensors)

            # Consistent with main tensor, only select r src tokens to be merged
            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, E_dim))
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, E_dim))

            dst_e = dst_e.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim), src_e_r, reduce=mode
            )
            if enable_protection:
                merged_extra_1 = torch.cat([unm_e, dst_e, protected_e], dim=1)
            else:
                merged_extra_1 = torch.cat([unm_e, dst_e], dim=1)

        if extra_tensors_2 is not None:
            E_dim_2 = extra_tensors_2.shape[-1]
            if enable_protection:
                src_e2, dst_e2, protected_e2 = split(extra_tensors_2)
            else:
                src_e2, dst_e2 = split(extra_tensors_2)

            src_e2_r = gather(src_e2, dim=-2, index=src_idx.expand(n, src_len, E_dim_2))
            unm_e2 = gather(src_e2, dim=-2, index=unm_idx.expand(n, unm_len, E_dim_2))

            dst_e2 = dst_e2.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim_2), src_e2_r, reduce=mode
            )
            if enable_protection:
                merged_extra_2 = torch.cat([unm_e2, dst_e2, protected_e2], dim=1)
            else:
                merged_extra_2 = torch.cat([unm_e2, dst_e2], dim=1)

        if enable_protection:
            main_result = torch.cat([unm, dst, protected], dim=1)
        else:
            main_result = torch.cat([unm, dst], dim=1)

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        elif merged_extra_1 is not None:
            return main_result, merged_extra_1
        else:
            return main_result

    # Define unmerge function to restore pre-merge state (for decoder)
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]

        if enable_protection:
            protected = x[
                ..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :
            ]

        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )

        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, src_len, c),
            src=src,
        )

        if enable_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(B, num_protected_actual, c),
                src=protected,
            )

        return out

    return merge, unmerge


def _uniform_frame_indices(
    num_imgs: int,
    protect_ratio: float,
    device: torch.device,
    explicit_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if explicit_indices is not None:
        indices = explicit_indices.to(device=device, dtype=torch.long)
        indices = indices[(indices >= 0) & (indices < num_imgs)]
        return torch.unique(indices, sorted=True)

    if num_imgs <= 0 or protect_ratio <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    num_protected = max(1, int(math.ceil(num_imgs * protect_ratio)))
    num_protected = min(num_protected, num_imgs)
    if num_protected == num_imgs:
        return torch.arange(num_imgs, device=device, dtype=torch.long)

    indices = torch.linspace(
        0,
        num_imgs - 1,
        steps=num_protected,
        device=device,
        dtype=torch.float32,
    ).round().long()

    indices = torch.unique(indices, sorted=True)
    if indices.numel() < num_protected:
        missing = num_protected - indices.numel()
        candidates = torch.arange(num_imgs, device=device, dtype=torch.long)
        mask = ~torch.isin(candidates, indices)
        indices = torch.sort(torch.cat([indices, candidates[mask][:missing]])).values
    return indices


def token_merge_frame_protected_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_frame_protection: bool = False,
    protected_frame_ratio: float = 0.1,
    protected_frame_indices: Optional[torch.Tensor] = None,
) -> Tuple[Callable, Callable]:
    """FastVGGT-style ToMe with full-frame protection.

    This is intentionally separate from ``fastvggt.merging.merge``. The original
    FastVGGT protection keeps a uniform 10% of tokens. Here, when protection is
    enabled, we keep a uniform 10% of frames, and all tokens from those frames are
    excluded from merging. The remaining frames use the same bipartite 2D merge
    pattern as the original implementation.
    """

    B, N, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather
    tokens_per_img = w * h + 5
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N, "Token count doesn't match (w*h+5)*num_imgs"

    with torch.no_grad():
        if enable_frame_protection:
            protected_frames = _uniform_frame_indices(
                num_imgs,
                protected_frame_ratio,
                metric.device,
                protected_frame_indices,
            )
            frame_offsets = protected_frames[:, None] * tokens_per_img
            token_offsets = torch.arange(tokens_per_img, device=metric.device)
            protected_indices = (frame_offsets + token_offsets[None, :]).reshape(-1)
        else:
            protected_frames = torch.empty(0, device=metric.device, dtype=torch.long)
            protected_indices = torch.empty(0, device=metric.device, dtype=torch.long)

        protected_mask = torch.zeros(N, device=metric.device, dtype=torch.bool)
        if protected_indices.numel() > 0:
            protected_mask[protected_indices] = True

        # -2: protected, -1: dst, 0: src. Protection has highest priority.
        idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx

        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

        if num_imgs > 1:
            cls_indices = (
                torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
            )
            cls_indices = cls_indices[:, None] + torch.arange(5, device=metric.device)
            idx_buffer_seq[cls_indices.flatten()] = -1

            effective_h = min(hsy * sy, h)
            effective_w = min(wsx * sx, w)
            effective_grid_size = effective_h * effective_w

            if no_rand:
                base_pattern = torch.zeros(
                    effective_grid_size, device=metric.device, dtype=torch.int64
                )
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img + 5
                )
                grid_indices = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=metric.device
                )
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                    num_imgs - 1
                )
            else:
                total_other_imgs = num_imgs - 1
                all_rand_idx = torch.randint(
                    sy * sx,
                    size=(total_other_imgs, hsy, wsx),
                    device=metric.device,
                    generator=generator,
                )

                scatter_src = -torch.ones(
                    total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64
                )
                idx_buffer_batch = torch.zeros(
                    total_other_imgs,
                    hsy,
                    wsx,
                    sy * sx,
                    device=metric.device,
                    dtype=torch.int64,
                )
                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )
                idx_buffer_batch = (
                    idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(total_other_imgs, hsy * sy, wsx * sx)
                )

                for i in range(total_other_imgs):
                    img_idx = i + 1
                    grid_start = img_idx * tokens_per_img + 5
                    flat_view = idx_buffer_batch[
                        i, :effective_h, :effective_w
                    ].flatten()
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = (
                        flat_view
                    )

        if protected_indices.numel() > 0:
            idx_buffer_seq[protected_indices] = -2

        all_idx = torch.arange(N, device=metric.device).reshape(1, -1, 1)
        a_idx = all_idx[:, idx_buffer_seq == 0, :]
        b_idx = all_idx[:, idx_buffer_seq == -1, :]
        protected_idx = all_idx[:, idx_buffer_seq == -2, :]

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]
        num_protected_actual = protected_idx.shape[1]

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            if enable_frame_protection:
                protected = gather(
                    x, dim=1, index=protected_idx.expand(B, num_protected_actual, C)
                )
                return src, dst, protected
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        if enable_frame_protection:
            a, b, _ = split(metric)
        else:
            a, b = split(metric)

        r = min(a.shape[1], r)
        num_src_actual = a.shape[1]
        if num_src_actual == 0 or num_dst == 0 or r <= 0:
            return do_nothing, do_nothing

        chunk_size = min(5000, num_src_actual)
        b_transposed = b.transpose(-1, -2)
        node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if enable_frame_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)

        n, _, c = src.shape
        unm_len = unm_idx.shape[1]
        src_len = src_idx.shape[1]

        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)

        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            e_dim = extra_tensors.shape[-1]
            if enable_frame_protection:
                src_e, dst_e, protected_e = split(extra_tensors)
            else:
                src_e, dst_e = split(extra_tensors)
            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, e_dim))
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, e_dim))
            dst_e = dst_e.scatter_reduce(
                -2, dst_idx.expand(n, src_len, e_dim), src_e_r, reduce=mode
            )
            if enable_frame_protection:
                merged_extra_1 = torch.cat([unm_e, dst_e, protected_e], dim=1)
            else:
                merged_extra_1 = torch.cat([unm_e, dst_e], dim=1)

        if extra_tensors_2 is not None:
            e_dim_2 = extra_tensors_2.shape[-1]
            if enable_frame_protection:
                src_e2, dst_e2, protected_e2 = split(extra_tensors_2)
            else:
                src_e2, dst_e2 = split(extra_tensors_2)
            src_e2_r = gather(src_e2, dim=-2, index=src_idx.expand(n, src_len, e_dim_2))
            unm_e2 = gather(src_e2, dim=-2, index=unm_idx.expand(n, unm_len, e_dim_2))
            dst_e2 = dst_e2.scatter_reduce(
                -2, dst_idx.expand(n, src_len, e_dim_2), src_e2_r, reduce=mode
            )
            if enable_frame_protection:
                merged_extra_2 = torch.cat([unm_e2, dst_e2, protected_e2], dim=1)
            else:
                merged_extra_2 = torch.cat([unm_e2, dst_e2], dim=1)

        if enable_frame_protection:
            main_result = torch.cat([unm, dst, protected], dim=1)
        else:
            main_result = torch.cat([unm, dst], dim=1)

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        if merged_extra_1 is not None:
            return main_result, merged_extra_1
        return main_result

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]

        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]
        if enable_frame_protection:
            protected = x[
                ..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :
            ]

        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, src_len, c),
            src=src,
        )

        if enable_frame_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(B, num_protected_actual, c),
                src=protected,
            )

        return out

    return merge, unmerge
