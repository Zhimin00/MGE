import numpy as np
from torch.utils.data import Sampler


class DynamicBatchedSampler(Sampler):
    """Pi3/VGGT-style dynamic batch sampler.

    Each yielded batch samples a random number of views `nview` in
    [min_view, max_view], and sets the number of sequences in the batch so
    that `batch_size * nview` is close to `target_images` (the per-GPU image
    budget). The number of batches per epoch is fixed to `iters_per_epoch`,
    decoupled from the underlying (weighted-resampled) dataset length.

    All ranks must build this sampler with the same arguments: the RNG is
    seeded purely from `epoch` (same convention as CustomRandomSampler in
    batched_sampler.py), so every process generates an identical sequence of
    batches; `accelerator.prepare(data_loader)` is what shards that sequence
    across ranks (round-robin over whole batches), same as the rest of this
    codebase's training scripts already rely on.
    """

    def __init__(
        self,
        dataset,
        target_images=64,
        min_view=2,
        max_view=24,
        iters_per_epoch=800,
    ):
        assert min_view >= 1 and max_view >= min_view
        self.len_dataset = len(dataset)
        self.pool_size = len(dataset._resolutions)
        self.target_images = target_images
        self.min_view = min_view
        self.max_view = max_view
        self.iters_per_epoch = iters_per_epoch
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.iters_per_epoch

    def __iter__(self):
        if self.epoch is None:
            raise ValueError(
                "Epoch number not set. Please call 'set_epoch(epoch)' before iterating."
            )

        rng = np.random.default_rng(seed=self.epoch + 788)

        pool = rng.permutation(self.len_dataset)
        ptr = 0

        for _ in range(self.iters_per_epoch):
            nview = int(rng.integers(self.min_view, self.max_view + 1))
            batch_size = max(1, round(self.target_images / nview))
            feat_idx = int(rng.integers(self.pool_size)) if self.pool_size > 1 else 0

            idxs = []
            for _ in range(batch_size):
                if ptr >= len(pool):
                    pool = rng.permutation(self.len_dataset)
                    ptr = 0
                idxs.append(int(pool[ptr]))
                ptr += 1

            yield [(idx, feat_idx, nview) for idx in idxs]
