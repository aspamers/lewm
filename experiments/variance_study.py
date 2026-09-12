"""Does a shape-free variance floor suffice for prediction and planning?"""
import argparse
from pathlib import Path

import torch

from regularizer_study import run

KINDS = ["none", "variance_floor", "variance_decorrelation", "gaussian"]
GRID = {"none": [0.], "variance_floor": [.3, 3., 30.],
        "variance_decorrelation": [.3, 3., 30.], "gaussian": [.03, .09, .27]}


@torch.no_grad()
def collapse_diagnostics(model, data):
    out = model.encode(dict(data))
    z = out["emb"].double()
    flat = z.flatten(0, 1)
    centered = flat-flat.mean(0)
    std = flat.std(0)
    standardized = centered/std.clamp_min(1e-8)
    corr = standardized.T@standardized/(len(flat)-1)
    d = flat.shape[-1]
    offdiag = corr-torch.diag_embed(corr.diagonal())
    eigen = torch.linalg.eigvalsh(corr).clamp_min(0)
    p = eigen/eigen.sum().clamp_min(1e-30)
    per_time_std = z.std(0)
    prediction = model.predict(out["emb"][:, :3], out["act_emb"][:, :3]).double()
    mse = (prediction-z[:, 1:4]).square().mean()
    return {"std_quantiles": torch.quantile(std, torch.tensor(
                [0., .1, .5, .9, 1.], dtype=std.dtype, device=std.device)).cpu().tolist(),
            "std_mean": float(std.mean()),
            "fraction_std_below_point1": float((std < .1).double().mean()),
            "fraction_std_at_least_point9": float((std >= .9).double().mean()),
            "timewise_fraction_std_at_least_point9": float((per_time_std >= .9).double().mean()),
            "mean_abs_offdiag_correlation": float(offdiag.abs().sum()/(d*(d-1))),
            "mean_squared_offdiag_correlation": float(offdiag.square().sum()/(d*(d-1))),
            "correlation_effective_rank": float((-(p*p.clamp_min(1e-30).log()).sum()).exp()),
            "correlation_top_eigen_fraction": float(p[-1]),
            "latent_prediction_mse": float(mse),
            "variance_normalized_prediction_mse": float(mse/std.square().mean().clamp_min(1e-8)),
            "mean_vector_squared_per_dim": float(flat.mean(0).square().mean())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(18, 23)))
    parser.add_argument("--output", type=Path, default=Path("outputs/variance-study-1000"))
    args = parser.parse_args()
    if args.steps < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("steps must be positive and seeds unique")
    if (args.output / "results.json").exists():
        parser.error("output already contains results; choose a new directory")
    run(args, grid=GRID, test_seed=72000, extra_diagnostics=collapse_diagnostics,
        sources=[Path(__file__)])
