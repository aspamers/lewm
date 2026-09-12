"""Episode-disjoint experiment partitions and training-only action statistics."""
import torch
from torch.utils.data import Subset


def split_episodes(dataset, train_fraction=0.9, seed=42):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    episodes = sorted({int(ep) for ep, _ in dataset.clip_indices})
    if len(episodes) < 2:
        raise ValueError("Need at least two episodes for a held-out split")
    order = torch.randperm(len(episodes), generator=torch.Generator().manual_seed(seed)).tolist()
    n = max(1, min(len(episodes)-1, int(len(episodes) * train_fraction)))
    train_eps = {episodes[i] for i in order[:n]}
    train_idx, val_idx = [], []
    for idx, (ep, _) in enumerate(dataset.clip_indices):
        (train_idx if int(ep) in train_eps else val_idx).append(idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx), sorted(train_eps)


def action_statistics(dataset, episodes):
    """Read cached actions only; exclude terminal NaNs and validation episodes."""
    data = torch.as_tensor(dataset.get_col_data("action"))
    total = torch.zeros(data.shape[-1], dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for ep in episodes:
        start = int(dataset.offsets[ep])
        x = data[start:start + int(dataset.lengths[ep])].double()
        x = x[torch.isfinite(x).all(dim=-1)]
        total += x.sum(0)
        squares += x.square().sum(0)
        count += len(x)
    if count < 2:
        raise ValueError("Insufficient finite training actions")
    mean = total / count
    std = ((squares - total.square() / count) / (count - 1)).clamp_min(1e-12).sqrt()
    return mean.float()[None], std.float()[None]
