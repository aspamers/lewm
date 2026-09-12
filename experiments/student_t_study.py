"""Covariance-matched Student-t vs Gaussian: fixed model/loss, no decoder."""
import argparse
from pathlib import Path

import torch

from regularizer_study import run


GRID = {kind: [.03, .09, .27] for kind in ("gaussian", "student_t3", "student_t5", "student_t9")}


@torch.no_grad()
def tail_diagnostics(model, data):
    """Fixed random projections, raw latent units: no per-model whitening.

    Measure spread as well as tails so collapse or excessive variance cannot
    masquerade as tail matching. Student-t(3) has no finite fourth moment;
    finite-sample kurtosis is intentionally not used as a pass/fail criterion.
    """
    z = model.encode(dict(data))["emb"].flatten(0, 1)
    gen = torch.Generator(device=z.device).manual_seed(11992)
    a = torch.randn(z.shape[-1], 64, generator=gen, device=z.device)
    a /= a.norm(dim=0)
    projected = (z @ a).flatten()
    absolute = projected.abs()
    return {"projection_abs_quantiles": torch.quantile(absolute, torch.tensor(
                [.5, .9, .95, .99], device=z.device)).cpu().tolist(),
            "projection_tail_gt3": float((absolute>3).float().mean()),
            "projection_tail_gt5": float((absolute>5).float().mean()),
            "projection_second_moment": float(projected.square().mean()),
            "projection_histogram": torch.histc(projected, bins=80, min=-8, max=8).cpu().tolist(),
            "projection_sample_count": projected.numel()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("outputs/student-t-study-1000"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    # Fresh test data: the previous study's final test has already been inspected.
    run(args, grid=GRID, test_seed=48000, extra_diagnostics=tail_diagnostics,
        sources=[Path("experiments/student_t_study.py")])
