"""Equal-budget radial shape sweep, preserving the projected SIGReg estimator."""
import argparse
from pathlib import Path

import torch

from regularizer_study import run
from student_t_study import tail_diagnostics

KINDS = ["radial_broad", "gaussian", "radial_half", "radial_quarter", "radial_sphere"]
GRID = {kind: [.03, .09, .27] for kind in KINDS}


@torch.no_grad()
def radial_diagnostics(model, data):
    z = model.encode(dict(data))["emb"].flatten(0, 1).double()
    # Raw radii about the target origin; shape CV also removes global scale.
    q = z.square().sum(-1) / z.shape[-1]
    centered = z - z.mean(0)
    qc = centered.square().sum(-1) / z.shape[-1]
    return {**tail_diagnostics(model, data),
            "radius_squared_mean_over_dim": float(q.mean()),
            "radius_squared_variance_over_dim_squared": float(q.var(unbiased=False)),
            "radius_squared_cv2": float(q.var(unbiased=False)/q.mean().square()),
            "centered_radius_squared_cv2": float(qc.var(unbiased=False)/qc.mean().square()),
            "mean_vector_squared_over_dim": float(z.mean(0).square().sum()/z.shape[-1]),
            "radius_over_sqrt_dim_quantiles": torch.quantile(q.sqrt(), torch.tensor(
                [.05, .25, .5, .75, .95], dtype=q.dtype, device=q.device)).cpu().tolist(),
            "radius_over_sqrt_dim_histogram": torch.histc(q.sqrt(), bins=60, min=0, max=3).cpu().tolist(),
            "radius_sample_count": len(q)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(13, 18)))
    parser.add_argument("--output", type=Path, default=Path("outputs/radial-study-1000"))
    args = parser.parse_args()
    if args.steps < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("steps must be positive and seeds unique")
    if (args.output / "results.json").exists():
        parser.error("output already contains results; choose a new directory")
    run(args, grid=GRID, test_seed=60000, extra_diagnostics=radial_diagnostics,
        sources=[Path(__file__), Path("experiments/student_t_study.py")], select_primary=True)
