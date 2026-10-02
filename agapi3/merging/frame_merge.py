import math
from typing import Callable, Optional, Tuple, Union

import torch

from .merge import do_nothing, fast_similarity_chunks


def _expand_idx(idx_1d: torch.Tensor, batch: int, channels: int) -> torch.Tensor:
    return idx_1d.view(1, -1, 1).expand(batch, -1, channels)


def legacy_original_retained_token_count(
    num_frames: int,
    patch_w: int,
    patch_h: int,
    sx: int,
    sy: int,
    merge_ratio: float,
    protected_token_ratio: float = 0.1,
) -> int:
    """Estimate the legacy original strategy's merged-attention sequence length."""
    hw = patch_w * patch_h
    tokens_per_frame = hw + 5
    total_tokens = num_frames * tokens_per_frame
    if num_frames <= 0 or total_tokens <= 0:
        return 0

    grid_destinations = (patch_h // sy) * (patch_w // sx)
    destination_tokens = tokens_per_frame
    destination_tokens += max(0, num_frames - 1) * (5 + grid_destinations)
    source_tokens = total_tokens - destination_tokens

    protected_tokens = int(total_tokens * protected_token_ratio)
    protected_sources = round(protected_tokens * source_tokens / total_tokens)
    valid_sources = max(0, source_tokens - protected_sources)
    actual_merges = min(int(total_tokens * merge_ratio), valid_sources)

    # The legacy implementation appends every protected token. Protected
    # destinations therefore also appear in the destination pool.
    protected_destinations = protected_tokens - protected_sources
    return total_tokens - actual_merges + protected_destinations


def _uniform_anchor_frames(
    num_frames: int,
    anchor_ratio: float,
    device: torch.device,
    keep_first: bool = True,
) -> torch.Tensor:
    if num_frames <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    num_anchor = int(math.ceil(num_frames * anchor_ratio))
    if keep_first:
        num_anchor = max(1, num_anchor)
    num_anchor = min(num_frames, max(0, num_anchor))
    if num_anchor == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if num_anchor == num_frames:
        return torch.arange(num_frames, device=device, dtype=torch.long)

    anchors = torch.linspace(
        0,
        num_frames - 1,
        steps=num_anchor,
        device=device,
        dtype=torch.float32,
    ).round().long()
    if keep_first:
        anchors[0] = 0
    anchors = torch.unique(anchors, sorted=True)
    if anchors.numel() < num_anchor:
        missing = num_anchor - anchors.numel()
        candidates = torch.arange(num_frames, device=device, dtype=torch.long)
        mask = ~torch.isin(candidates, anchors)
        anchors = torch.sort(torch.cat([anchors, candidates[mask][:missing]])).values
    return anchors


def _diverse_anchor_frames(
    frame_desc: torch.Tensor,
    anchor_ratio: float,
    keep_first: bool = True,
) -> torch.Tensor:
    num_frames = frame_desc.shape[0]
    device = frame_desc.device
    if num_frames <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    num_anchor = int(math.ceil(num_frames * anchor_ratio))
    if keep_first:
        num_anchor = max(1, num_anchor)
    num_anchor = min(num_frames, max(0, num_anchor))
    if num_anchor == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if num_anchor == num_frames:
        return torch.arange(num_frames, device=device, dtype=torch.long)

    desc = frame_desc.float()
    desc = desc / desc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    similarity = torch.mm(desc, desc.transpose(0, 1)).clamp(0.0, 1.0)

    # DA-VGGT-style FPS in cosine-similarity space: choose the frame whose
    # maximum similarity to the selected set is smallest.
    anchors = []
    if keep_first:
        anchors.append(0)
    else:
        anchors.append(int(desc.norm(dim=-1).argmax().item()))

    selected = torch.zeros(num_frames, device=device, dtype=torch.bool)
    selected[anchors[0]] = True
    min_sim_to_anchor = similarity[:, anchors[0]]

    while len(anchors) < num_anchor:
        scores = min_sim_to_anchor.masked_fill(selected, float("inf"))
        next_anchor = int(scores.argmin().item())
        anchors.append(next_anchor)
        selected[next_anchor] = True
        min_sim_to_anchor = torch.maximum(min_sim_to_anchor, similarity[:, next_anchor])

    return torch.tensor(sorted(anchors), device=device, dtype=torch.long)


def _da_partition_anchor_frames(
    frame_desc: torch.Tensor,
    anchor_ratio: float,
    chunk_size: int = 50,
    local_search_iters: int = 5,
    seed: int = 42,
) -> torch.Tensor:
    """Select anchors from DA-VGGT-style diversity-aware graph partitions.

    The partition uses the official random-balanced initialization followed by
    2-opt refinement on reverse cosine similarity. Anchors are then selected
    with the official coverage-times-diversity criterion over the partitions.
    """
    num_frames = int(frame_desc.shape[0])
    device = frame_desc.device
    if num_frames <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    num_anchor = min(
        num_frames,
        max(1, int(math.ceil(num_frames * anchor_ratio))),
    )
    desc = frame_desc.detach().float()
    desc = desc / desc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = torch.mm(desc, desc.transpose(0, 1)).clamp(0.0, 1.0)

    num_chunks = max(1, num_frames // max(1, int(chunk_size)))
    chunks = [[] for _ in range(num_chunks)]
    chunk_counts = [0] * num_chunks
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for frame in torch.randperm(num_frames, generator=generator).tolist():
        chunk = min(range(num_chunks), key=chunk_counts.__getitem__)
        chunks[chunk].append(frame)
        chunk_counts[chunk] += 1

    utility = 1.0 - sim.to(torch.float64)
    utility.fill_diagonal_(0.0)
    chunk_utility = torch.zeros(
        num_chunks,
        num_frames,
        device=device,
        dtype=torch.float64,
    )
    for chunk, indices in enumerate(chunks):
        chunk_utility[chunk] = utility[:, indices].sum(dim=1)

    for _ in range(max(0, int(local_search_iters))):
        improved = False
        for first in range(num_chunks):
            for second in range(first + 1, num_chunks):
                while True:
                    first_frames = torch.as_tensor(chunks[first], device=device)
                    second_frames = torch.as_tensor(chunks[second], device=device)
                    if first_frames.numel() == 0 or second_frames.numel() == 0:
                        break
                    row = chunk_utility[second][first_frames] - chunk_utility[first][first_frames]
                    col = chunk_utility[first][second_frames] - chunk_utility[second][second_frames]
                    delta = (
                        row[:, None]
                        + col[None, :]
                        - 2.0 * utility[first_frames[:, None], second_frames[None, :]]
                    )
                    best_flat = int(delta.argmax().item())
                    row_idx, col_idx = divmod(best_flat, second_frames.numel())
                    if delta[row_idx, col_idx] <= 1e-8:
                        break
                    first_frame = int(first_frames[row_idx])
                    second_frame = int(second_frames[col_idx])
                    chunks[first][row_idx] = second_frame
                    chunks[second][col_idx] = first_frame
                    chunk_utility[first] += utility[:, second_frame] - utility[:, first_frame]
                    chunk_utility[second] += utility[:, first_frame] - utility[:, second_frame]
                    improved = True
        if not improved:
            break

    chunk_coverage = torch.stack(
        [sim[indices].max(dim=0).values for indices in chunks],
        dim=0,
    )
    anchor_scores = chunk_coverage.min(dim=0).values
    anchors = []
    selected = torch.zeros(num_frames, device=device, dtype=torch.bool)
    for index in range(num_anchor):
        if index == 0:
            best = int(anchor_scores.argmax().item())
        else:
            similarity_to_selected = sim[:, anchors].max(dim=1).values
            scores = anchor_scores * (1.0 - similarity_to_selected)
            scores[selected] = -1.0
            best = int(scores.argmax().item())
        anchors.append(best)
        selected[best] = True

    return torch.tensor(anchors, device=device, dtype=torch.long)


def _patch_salience(patch_tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    B, num_frames, hw, C = patch_tokens.shape
    patch_grid = patch_tokens.float().reshape(B, num_frames, patch_h, patch_w, C)
    center = patch_grid - patch_grid.mean(dim=(2, 3), keepdim=True)
    salience = center.square().mean(dim=-1)
    if patch_h > 1:
        diff_h = (patch_grid[:, :, 1:, :, :] - patch_grid[:, :, :-1, :, :]).square().mean(dim=-1)
        salience[:, :, 1:, :] = salience[:, :, 1:, :] + diff_h
        salience[:, :, :-1, :] = salience[:, :, :-1, :] + diff_h
    if patch_w > 1:
        diff_w = (patch_grid[:, :, :, 1:, :] - patch_grid[:, :, :, :-1, :]).square().mean(dim=-1)
        salience[:, :, :, 1:] = salience[:, :, :, 1:] + diff_w
        salience[:, :, :, :-1] = salience[:, :, :, :-1] + diff_w

    return salience.mean(dim=0).reshape(num_frames, hw)


def token_merge_pi3_geometry_aware_global_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    merge_ratio: float,
    generator: Optional[torch.Generator] = None,
    anchor_frame_ratio: float = 0.1,
    keep_first_anchor: bool = True,
    protected_patch_ratio: float = 0.05,
    min_non_anchor_merge_ratio: float = 0.15,
) -> Tuple[Callable, Callable]:
    """Pi3-oriented global merge.

    Registers and diverse anchor frames stay exact. Non-anchor patch tokens are
    merged patch-to-patch only, with per-frame budgets increased for frames that
    are already well represented by the anchors.
    """

    B, N, _ = metric.shape
    if merge_ratio <= 0:
        return do_nothing, do_nothing

    gather = torch.gather
    num_register = 5
    hw = w * h
    tokens_per_frame = hw + num_register
    num_frames = N // tokens_per_frame
    assert tokens_per_frame * num_frames == N, "Token count doesn't match (w*h+5)*num_frames"
    if num_frames <= 0 or hw <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric_norm = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        frame_tokens = metric_norm.reshape(B, num_frames, tokens_per_frame, -1)
        patch_tokens = frame_tokens[:, :, num_register:, :]
        frame_desc = patch_tokens.mean(dim=2).mean(dim=0)
        anchors = _diverse_anchor_frames(
            frame_desc,
            anchor_frame_ratio,
            keep_first=keep_first_anchor,
        )
        is_anchor_frame = torch.zeros(num_frames, device=metric.device, dtype=torch.bool)
        if anchors.numel() > 0:
            is_anchor_frame[anchors] = True

        salience = _patch_salience(patch_tokens, h, w)
        protected_patch_count = int(math.ceil(hw * protected_patch_ratio))
        protected_patch_count = min(hw, max(0, protected_patch_count))

        token_role = torch.full((N,), -2, device=metric.device, dtype=torch.int64)
        hsy, wsx = max(1, h // sy), max(1, w // sx)
        effective_h = min(hsy * sy, h)
        effective_w = min(wsx * sx, w)

        non_anchor_frames = torch.nonzero(~is_anchor_frame, as_tuple=False).flatten()
        if non_anchor_frames.numel() == 0:
            return do_nothing, do_nothing

        for frame in non_anchor_frames.tolist():
            patch_start = frame * tokens_per_frame + num_register
            role_patch = torch.zeros(hw, device=metric.device, dtype=torch.int64)

            if protected_patch_count > 0:
                protected_local = salience[frame].topk(protected_patch_count).indices
                role_patch[protected_local] = -2

            for gy in range(hsy):
                for gx in range(wsx):
                    rows = torch.arange(gy * sy, min((gy + 1) * sy, effective_h), device=metric.device)
                    cols = torch.arange(gx * sx, min((gx + 1) * sx, effective_w), device=metric.device)
                    if rows.numel() == 0 or cols.numel() == 0:
                        continue
                    local = (rows[:, None] * w + cols[None, :]).reshape(-1)
                    available = local[role_patch[local] != -2]
                    if available.numel() == 0:
                        continue
                    dst_local = available[salience[frame, available].argmax()]
                    role_patch[dst_local] = -1

            token_role[patch_start : patch_start + hw] = role_patch

        all_idx = torch.arange(N, device=metric.device).reshape(1, -1, 1)
        a_idx = all_idx[:, token_role == 0, :]
        b_idx = all_idx[:, token_role == -1, :]
        protected_idx = all_idx[:, token_role == -2, :]

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]
        num_protected = protected_idx.shape[1]
        if num_src == 0 or num_dst == 0:
            return do_nothing, do_nothing

        src_global = a_idx[0, :, 0]
        src_frame = src_global // tokens_per_frame
        dst_global = b_idx[0, :, 0]
        protected_global = protected_idx[0, :, 0]

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            protected = gather(x, dim=1, index=protected_idx.expand(B, num_protected, C))
            return src, dst, protected

        anchor_desc = frame_desc[anchors] if anchors.numel() > 0 else frame_desc[:1]
        frame_desc_n = frame_desc.float() / frame_desc.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        anchor_desc_n = anchor_desc.float() / anchor_desc.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sim_to_anchor = torch.mm(frame_desc_n, anchor_desc_n.transpose(0, 1)).max(dim=1).values
        non_anchor_sim = sim_to_anchor[non_anchor_frames]
        if non_anchor_sim.numel() > 1:
            sim_norm = (sim_to_anchor - non_anchor_sim.min()) / (non_anchor_sim.max() - non_anchor_sim.min()).clamp(min=1e-6)
        else:
            sim_norm = torch.ones_like(sim_to_anchor)
        low = min(float(merge_ratio), float(min_non_anchor_merge_ratio))
        high = max(float(merge_ratio), float(min_non_anchor_merge_ratio))
        ratios = low + (high - low) * sim_norm.float()
        ratios[is_anchor_frame] = 0.0

        src, dst, _ = split(metric_norm)
        chunk_size = min(5000, num_src)
        node_max, node_idx = fast_similarity_chunks(src, dst.transpose(-1, -2), chunk_size)
        scores_for_order = node_max.mean(dim=0)

        merge_src_positions = []
        keep_src_positions = []
        for frame in range(num_frames):
            frame_positions = torch.nonzero(src_frame == frame, as_tuple=False).flatten()
            if frame_positions.numel() == 0:
                continue
            frame_order = frame_positions[
                scores_for_order[frame_positions].argsort(descending=True)
            ]
            r_frame = min(int(hw * float(ratios[frame].item())), frame_order.numel())
            if r_frame > 0:
                merge_src_positions.append(frame_order[:r_frame])
            if r_frame < frame_order.numel():
                keep_src_positions.append(frame_order[r_frame:])

        if not merge_src_positions:
            return do_nothing, do_nothing

        src_pos = torch.cat(merge_src_positions, dim=0).view(1, -1, 1)
        if keep_src_positions:
            unm_idx = torch.cat(keep_src_positions, dim=0).view(1, -1, 1)
        else:
            unm_idx = torch.empty(1, 0, 1, device=metric.device, dtype=torch.long)
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_pos.expand(B, -1, -1))

        src_global_merge = gather(src_global.view(1, -1), dim=1, index=src_pos[:, :, 0]).view(-1)
        src_global_keep = gather(src_global.view(1, -1), dim=1, index=unm_idx[:, :, 0]).view(-1)

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
        def merge_one(t: torch.Tensor) -> torch.Tensor:
            batch = t.shape[0]
            channels = t.shape[2]
            src, dst, protected = split(t)
            src_keep = gather(src, dim=-2, index=unm_idx.expand(batch, unm_idx.shape[1], channels))
            src_merge = gather(src, dim=-2, index=src_pos.expand(batch, src_pos.shape[1], channels))
            dst = dst.scatter_reduce(
                -2,
                dst_idx.expand(batch, src_pos.shape[1], channels),
                src_merge,
                reduce=mode,
            )
            return torch.cat([src_keep, dst, protected], dim=1)

        out = merge_one(x)
        extra_1 = merge_one(extra_tensors) if extra_tensors is not None else None
        extra_2 = merge_one(extra_tensors_2) if extra_tensors_2 is not None else None
        if extra_1 is not None and extra_2 is not None:
            return out, extra_1, extra_2
        if extra_1 is not None:
            return out, extra_1
        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        channels = x.shape[2]
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        protected_len = num_protected
        src_keep = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]
        protected = x[..., unm_len + dst_len : unm_len + dst_len + protected_len, :]
        src_merged = gather(dst, dim=-2, index=dst_idx.expand(batch, src_pos.shape[1], channels))
        out = torch.zeros(batch, N, channels, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=_expand_idx(dst_global, batch, channels), src=dst)
        out.scatter_(dim=-2, index=_expand_idx(protected_global, batch, channels), src=protected)
        out.scatter_(dim=-2, index=_expand_idx(src_global_keep, batch, channels), src=src_keep)
        out.scatter_(dim=-2, index=_expand_idx(src_global_merge, batch, channels), src=src_merged)
        return out

    return merge, unmerge


def token_merge_frame_adaptive_global_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    anchor_frame_ratio: float = 0.1,
    anchor_merge_ratio: float = 0.0,
    non_anchor_merge_ratio: Optional[float] = None,
    keep_first_anchor: bool = True,
    adaptive_by_anchor_similarity: bool = False,
    min_non_anchor_merge_ratio: float = 0.3,
    max_non_anchor_merge_ratio: float = 0.75,
    target_total_merge_count: Optional[int] = None,
    anchor_selection: str = "da_partition",
    anchor_indices: Optional[torch.Tensor] = None,
) -> Tuple[Callable, Callable]:
    """Global ToMe with different merge quotas for different frames.

    Compared with frame-protected global merge, this keeps cross-frame matching
    but controls each frame's merge budget. Anchor frames get a low merge budget.
    Non-anchor frames can either share a fixed high budget or adapt their budget
    by similarity to anchors: redundant frames merge more, distinctive frames
    merge less.
    """

    B, N, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather
    tokens_per_frame = w * h + 5
    num_frames = N // tokens_per_frame
    assert tokens_per_frame * num_frames == N, "Token count doesn't match (w*h+5)*num_frames"

    hw = w * h
    if non_anchor_merge_ratio is None:
        non_anchor_merge_ratio = max(0.0, min(1.0, float(r) / max(1, N)))

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if anchor_indices is not None:
            anchors = anchor_indices.to(device=metric.device, dtype=torch.long)
        elif anchor_selection == "da_partition":
            frame_tokens = metric.reshape(B, num_frames, tokens_per_frame, -1)
            frame_desc = frame_tokens[:, :, 5:, :].mean(dim=2).mean(dim=0)
            anchors = _da_partition_anchor_frames(frame_desc, anchor_frame_ratio)
        elif anchor_selection == "fps_sim":
            frame_tokens = metric.reshape(B, num_frames, tokens_per_frame, -1)
            frame_desc = frame_tokens[:, :, 5:, :].mean(dim=2).mean(dim=0)
            anchors = _diverse_anchor_frames(
                frame_desc,
                anchor_frame_ratio,
                keep_first=keep_first_anchor,
            )
        elif anchor_selection == "uniform":
            anchors = _uniform_anchor_frames(
                num_frames,
                anchor_frame_ratio,
                metric.device,
                keep_first=keep_first_anchor,
            )
        else:
            raise ValueError(
                f"Unknown anchor selection: {anchor_selection!r} "
                "(expected 'da_partition', 'fps_sim', or 'uniform')"
            )
        is_anchor_frame = torch.zeros(num_frames, device=metric.device, dtype=torch.bool)
        if anchors.numel() > 0:
            is_anchor_frame[anchors] = True

        idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx

        # Register tokens are always destinations. Patch destinations are sampled
        # per frame, so all frames remain possible cross-frame merge targets.
        cls_indices = torch.arange(num_frames, device=metric.device) * tokens_per_frame
        cls_indices = cls_indices[:, None] + torch.arange(5, device=metric.device)
        idx_buffer_seq[cls_indices.flatten()] = -1

        effective_h = min(hsy * sy, h)
        effective_w = min(wsx * sx, w)
        effective_grid_size = effective_h * effective_w
        if num_frames > 0:
            if no_rand:
                base_pattern = torch.zeros(
                    effective_grid_size, device=metric.device, dtype=torch.int64
                )
                for frame in range(num_frames):
                    grid_start = frame * tokens_per_frame + 5
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = base_pattern
            else:
                all_rand_idx = torch.randint(
                    sy * sx,
                    size=(num_frames, hsy, wsx),
                    device=metric.device,
                    generator=generator,
                )
                scatter_src = -torch.ones(
                    num_frames, hsy, wsx, device=metric.device, dtype=torch.int64
                )
                idx_buffer_batch = torch.zeros(
                    num_frames,
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
                    idx_buffer_batch.view(num_frames, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(num_frames, hsy * sy, wsx * sx)
                )
                for frame in range(num_frames):
                    grid_start = frame * tokens_per_frame + 5
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = (
                        idx_buffer_batch[frame, :effective_h, :effective_w].flatten()
                    )

        all_idx = torch.arange(N, device=metric.device).reshape(1, -1, 1)
        a_idx = all_idx[:, idx_buffer_seq == 0, :]
        b_idx = all_idx[:, idx_buffer_seq == -1, :]

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]
        if num_src == 0 or num_dst == 0:
            return do_nothing, do_nothing

        src_global = a_idx[0, :, 0]
        src_frame = src_global // tokens_per_frame
        source_capacity = torch.bincount(src_frame, minlength=num_frames)
        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        sim_to_anchor = None
        if adaptive_by_anchor_similarity and anchors.numel() > 0:
            frame_tokens = metric.reshape(B, num_frames, tokens_per_frame, -1)
            frame_desc = frame_tokens[:, :, 5:, :].mean(dim=2).mean(dim=0)
            frame_desc = frame_desc / frame_desc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            anchor_desc = frame_desc[anchors]
            sim_to_anchor = torch.mm(frame_desc, anchor_desc.transpose(0, 1)).max(dim=1).values

            non_anchor_mask = ~is_anchor_frame
            ratios = torch.full(
                (num_frames,),
                float(anchor_merge_ratio),
                device=metric.device,
                dtype=torch.float32,
            )
            if non_anchor_mask.any():
                non_anchor_sim = sim_to_anchor[non_anchor_mask].float()
                if non_anchor_sim.numel() > 1:
                    sim_min = non_anchor_sim.min()
                    sim_max = non_anchor_sim.max()
                    sim_norm = (non_anchor_sim - sim_min) / (sim_max - sim_min).clamp(min=1e-6)
                else:
                    sim_norm = torch.ones_like(non_anchor_sim)
                low = min(float(min_non_anchor_merge_ratio), float(max_non_anchor_merge_ratio))
                high = min(
                    max(float(non_anchor_merge_ratio), low),
                    float(max_non_anchor_merge_ratio),
                )
                ratios[non_anchor_mask] = low + (high - low) * sim_norm
            if target_total_merge_count is not None and non_anchor_mask.any():
                capacity = source_capacity[non_anchor_mask]
                max_count_per_frame = torch.minimum(
                    torch.as_tensor(int(hw * high), device=metric.device, dtype=torch.long),
                    capacity.min(),
                )
                max_target = max_count_per_frame * non_anchor_sim.numel()
                target = torch.as_tensor(
                    target_total_merge_count,
                    device=metric.device,
                    dtype=torch.long,
                ).clamp_min(0)
                target = torch.minimum(target, max_target)

                target_mean = target.to(torch.float64) / (non_anchor_sim.numel() * hw)
                centered = sim_norm.to(torch.float64) - sim_norm.to(torch.float64).mean()
                lower_bound = torch.minimum(
                    target_mean,
                    torch.as_tensor(low, device=metric.device, dtype=torch.float64),
                )
                upper_bound = torch.as_tensor(
                    max_count_per_frame / hw,
                    device=metric.device,
                    dtype=torch.float64,
                )
                positive_extent = centered.clamp_min(0).max()
                negative_extent = (-centered).clamp_min(0).max()
                infinity = torch.full((), float("inf"), device=metric.device, dtype=torch.float64)
                alpha_high = torch.where(
                    positive_extent > 0,
                    (upper_bound - target_mean) / positive_extent,
                    infinity,
                )
                alpha_low = torch.where(
                    negative_extent > 0,
                    (target_mean - lower_bound) / negative_extent,
                    infinity,
                )
                alpha = torch.minimum(alpha_high, alpha_low).clamp_min(0)
                alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
                calibrated = target_mean + alpha * centered

                float_counts = calibrated * hw
                non_anchor_counts = float_counts.floor().long()
                remainder = target - non_anchor_counts.sum()
                fractional_rank = float_counts.frac().argsort(descending=True).argsort()
                non_anchor_counts += (fractional_rank < remainder).long()
                ratios[non_anchor_mask] = calibrated.to(ratios.dtype)

            r_by_frame = (ratios * hw).long()
            if target_total_merge_count is not None and non_anchor_mask.any():
                r_by_frame[non_anchor_mask] = non_anchor_counts
        else:
            r_by_frame = torch.where(
                is_anchor_frame,
                torch.full((num_frames,), int(hw * anchor_merge_ratio), device=metric.device, dtype=torch.long),
                torch.full((num_frames,), int(hw * non_anchor_merge_ratio), device=metric.device, dtype=torch.long),
            )
            non_anchor_mask = ~is_anchor_frame
            if target_total_merge_count is not None and non_anchor_mask.any():
                capacities = source_capacity[non_anchor_mask]
                target = torch.as_tensor(
                    target_total_merge_count,
                    device=metric.device,
                    dtype=torch.long,
                ).clamp(min=0, max=int(capacities.sum().item()))
                num_non_anchor = int(non_anchor_mask.sum().item())
                fixed_count = target // num_non_anchor
                non_anchor_counts = torch.minimum(
                    torch.full_like(capacities, fixed_count),
                    capacities,
                )
                remainder = target - non_anchor_counts.sum()
                eligible_rank = (capacities > non_anchor_counts).long().cumsum(dim=0) - 1
                non_anchor_counts += (
                    (capacities > non_anchor_counts) & (eligible_rank < remainder)
                ).long()
                r_by_frame[non_anchor_mask] = non_anchor_counts

        r_by_frame = torch.minimum(r_by_frame, source_capacity)

        a, b = split(metric)
        chunk_size = min(5000, num_src)
        node_max, node_idx = fast_similarity_chunks(a, b.transpose(-1, -2), chunk_size)

        merge_src_positions = []
        keep_src_positions = []
        scores_for_order = node_max.mean(dim=0)
        for frame in range(num_frames):
            frame_positions = torch.nonzero(src_frame == frame, as_tuple=False).flatten()
            if frame_positions.numel() == 0:
                continue
            frame_order = frame_positions[
                scores_for_order[frame_positions].argsort(descending=True)
            ]
            r_frame = min(int(r_by_frame[frame].item()), frame_order.numel())
            if r_frame > 0:
                merge_src_positions.append(frame_order[:r_frame])
            if r_frame < frame_order.numel():
                keep_src_positions.append(frame_order[r_frame:])

        if not merge_src_positions:
            return do_nothing, do_nothing

        src_pos = torch.cat(merge_src_positions, dim=0).view(1, -1, 1)
        if keep_src_positions:
            unm_idx = torch.cat(keep_src_positions, dim=0).view(1, -1, 1)
        else:
            unm_idx = torch.empty(1, 0, 1, device=metric.device, dtype=torch.long)
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_pos.expand(B, -1, -1))

        src_global_merge = gather(src_global.view(1, -1), dim=1, index=src_pos[:, :, 0]).view(-1)
        src_global_keep = gather(src_global.view(1, -1), dim=1, index=unm_idx[:, :, 0]).view(-1)
        dst_global = b_idx[0, :, 0]

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
        def merge_one(t: torch.Tensor) -> torch.Tensor:
            batch = t.shape[0]
            channels = t.shape[2]
            src, dst = split(t)
            src_keep = gather(src, dim=-2, index=unm_idx.expand(batch, unm_idx.shape[1], channels))
            src_merge = gather(src, dim=-2, index=src_pos.expand(batch, src_pos.shape[1], channels))
            dst = dst.scatter_reduce(
                -2,
                dst_idx.expand(batch, src_pos.shape[1], channels),
                src_merge,
                reduce=mode,
            )
            return torch.cat([src_keep, dst], dim=1)

        out = merge_one(x)
        extra_1 = merge_one(extra_tensors) if extra_tensors is not None else None
        extra_2 = merge_one(extra_tensors_2) if extra_tensors_2 is not None else None
        if extra_1 is not None and extra_2 is not None:
            return out, extra_1, extra_2
        if extra_1 is not None:
            return out, extra_1
        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        channels = x.shape[2]
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_keep = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]
        src_merged = gather(dst, dim=-2, index=dst_idx.expand(batch, src_pos.shape[1], channels))
        out = torch.zeros(batch, N, channels, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=_expand_idx(dst_global, batch, channels), src=dst)
        out.scatter_(dim=-2, index=_expand_idx(src_global_keep, batch, channels), src=src_keep)
        out.scatter_(dim=-2, index=_expand_idx(src_global_merge, batch, channels), src=src_merged)
        return out

    merge.anchor_frames = anchors.detach().clone()
    merge.merge_counts = r_by_frame.detach().clone()
    merge.frame_similarity = (
        sim_to_anchor.detach().clone() if sim_to_anchor is not None else None
    )
    merge.target_total_merge_count = target_total_merge_count

    return merge, unmerge
