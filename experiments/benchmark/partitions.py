"""Deterministic episode partitions and goals, independent of model outcomes."""
import numpy as np
import torch


class EvaluationSeeds:
    """Supply fixed reset seeds when archives omit them; never label them original."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.injected = "seed" not in dataset.column_names

    @property
    def column_names(self):
        return [*self.dataset.column_names, *(["seed"] if self.injected else [])]

    def load_chunk(self, episodes_idx, start, end):
        chunks = self.dataset.load_chunk(episodes_idx, start, end)
        if self.injected:
            for ep, chunk in zip(episodes_idx, chunks):
                chunk["seed"] = torch.full((len(chunk["pixels"]),), 6173+int(ep), dtype=torch.int64)
        return chunks


def make_partition(lengths, seed=9031, goals=16):
    lengths = np.asarray(lengths)
    eligible = np.flatnonzero(lengths >= 30)
    if len(eligible) < 30:
        raise ValueError("Need at least 30 sufficiently long episodes")
    rng = np.random.default_rng(seed)
    order = rng.permutation(eligible)
    n_hold = max(1, len(order)//10)
    splits = {"train": order[2*n_hold:], "validation": order[:n_hold], "test": order[n_hold:2*n_hold]}
    result = {key: sorted(value.tolist()) for key, value in splits.items()}
    for key in ("validation", "test"):
        # Repeated episodes permitted only if the held-out split is smaller
        # than the fixed task budget. Repeated tasks are not independent trials.
        chosen = rng.choice(splits[key], size=goals, replace=len(splits[key]) < goals)
        starts = [int(rng.integers(0, lengths[ep]-25)) for ep in chosen]
        result[key+"_goals"] = {"episodes": chosen.tolist(), "starts": starts}
    result["seed"] = seed
    return result
