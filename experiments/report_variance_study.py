"""Compare shape-free collapse prevention with SIGReg on frozen test tasks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from variance_study import KINDS

LABELS = ["None", "Variance floor", "Floor + decorrelation", "SIGReg"]


def interval(values):
    rng = np.random.default_rng(20260912)
    return np.quantile(values[rng.integers(len(values), size=(20000, len(values)))].mean(1), [.025, .975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "results.json").read_text())
    seeds = report["seeds"]
    expected = len(seeds)*sum(len(w) for w in report["grid"].values())
    if len(report["validation"]) != expected or len(report["test"]) != len(seeds)*4:
        raise ValueError("Wait for the full sweep and test evaluation")
    lookup = {(r["seed"], r["kind"]): r for r in report["test"]}
    assert len(lookup) == len(report["test"])
    for r in report["test"]:
        distances = np.array(r["per_episode_distance"])
        assert len(distances) == 24 and np.isfinite(distances).all()
        assert np.isclose(distances.mean(), r["goal_distance"], atol=1e-7)
        assert np.isclose((distances < .12).mean(), r["success_rate"], atol=1e-7)
        assert r["weight"] == report["selected_weights"][r["kind"]]
    def matrix(metric):
        return np.array([[lookup[s, k][metric] for k in KINDS] for s in seeds])
    distance = matrix("goal_distance")
    success = matrix("success_rate")
    contrasts = {}
    for name, left, right in [("Floor minus SIGReg (primary)", 1, 3),
                              ("Combined minus floor", 2, 1),
                              ("Combined minus SIGReg", 2, 3),
                              ("Floor minus none", 1, 0),
                              ("Combined minus none", 2, 0),
                              ("SIGReg minus none", 3, 0)]:
        values = distance[:, left]-distance[:, right]
        contrasts[name] = {"mean": float(values.mean()), "bootstrap_95_interval": interval(values).tolist(),
                           "per_seed": values.tolist()}
    boundary = [k for k in KINDS if len(report["grid"][k]) > 1 and
                report["selected_weights"][k] in (min(report["grid"][k]), max(report["grid"][k]))]
    lines = ["# Variance floor versus SIGReg", "", "**Synthetic pilot, not published LeWM/TwoRoom.**", "",
             f"{expected} runs, {report['steps']} updates each, seeds {seeds}. Three coefficients "
             "per active regularizer, plus no-regularizer control. Each arm's coefficient was "
             "selected by mean validation planning distance before fresh test seed 72000 was opened.", "",
             "## Planning and physical prediction", "",
             "Mean +/- sample standard deviation across five training seeds. Lower errors are better.", "",
             "| Arm | Weight | Goal distance | Success | Position probe MSE | Five-step position probe MSE |",
             "|---|---|---|---|---|---|"]
    for i, kind in enumerate(KINDS):
        lines.append(f"| {LABELS[i]} | {report['selected_weights'][kind]} | "
                     f"{distance[:, i].mean():.5f} +/- {distance[:, i].std(ddof=1):.5f} | "
                     f"{success[:, i].mean():.1%} +/- {success[:, i].std(ddof=1)*100:.1f} pp | "
                     f"{matrix('position_probe_mse')[:, i].mean():.5g} | "
                     f"{matrix('future_position_probe_mse')[:, i].mean():.5g} |")
    first = report["test"][0]
    lines += ["", f"Controls: stay-put mean distance {first['stay_distance']:.5f}; random plan "
              f"{first['random_distance']:.5f}.", "", "![Planning and collapse diagnostics](variance-comparison.png)", "",
              "## Are dimensions active and distinct?", "",
              "The floor threshold is std=1. We report the fraction reaching at least 0.9 "
              "as a practical near-floor diagnostic, not proof that the soft constraint is "
              "fully satisfied. Standard deviations and correlations pool test frames; "
              "timewise attainment is also saved in raw results.", "",
              "| Arm | Mean coordinate std | Fraction std <0.1 | Fraction std >=0.9 | Total variance | Covariance effective rank | Correlation effective rank | Mean abs correlation | Top correlation eigenvalue share |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i in range(4):
        def avg(key):
            return matrix(key)[:, i].mean()
        lines.append(f"| {LABELS[i]} | {avg('std_mean'):.4f} | {avg('fraction_std_below_point1'):.1%} | "
                     f"{avg('fraction_std_at_least_point9'):.1%} | {avg('latent_variance'):.5g} | "
                     f"{avg('effective_rank'):.2f} | {avg('correlation_effective_rank'):.2f} | "
                     f"{avg('mean_abs_offdiag_correlation'):.4f} | {avg('correlation_top_eigen_fraction'):.1%} |")
    lines += ["", "Effective rank is a relative eigenvalue-spread measure: high rank at nearly "
              "zero total variance still represents near-collapse. Conversely, high coordinate "
              "variance with low rank can indicate repeated signals. Neither statistic alone "
              "measures useful information; physical probes and planning are the outcome checks.", "",
              "## Latent prediction diagnostics", "",
              "Raw latent MSE can become tiny through collapse. The normalized metric divides "
              "by mean coordinate variance (clamped at 1e-8); interpret it with physical probes.", "",
              "| Arm | Raw one-step MSE | Variance-normalized MSE | Action-shuffle five-step probe degradation | Mean-vector squared / D |",
              "|---|---|---|---|---|"]
    for i in range(4):
        lines.append(f"| {LABELS[i]} | {matrix('latent_prediction_mse')[:, i].mean():.5g} | "
                     f"{matrix('variance_normalized_prediction_mse')[:, i].mean():.5g} | "
                     f"{matrix('action_shuffle_probe_delta')[:, i].mean():.5g} | "
                     f"{matrix('mean_vector_squared_per_dim')[:, i].mean():.5g} |")
    lines += ["", "## Paired goal-distance contrasts", "",
              "Negative favors the first arm. 20,000 paired training-seed bootstrap resamples, "
              "RNG seed 20260912. Five seeds remain a small sample. Intervals condition on "
              "fixed data and selected coefficients; no dataset or search uncertainty is "
              "included. Only floor versus SIGReg is primary; other intervals are exploratory "
              "without multiple-comparison correction.", "",
              "| Contrast | Mean difference | 95% seed bootstrap interval |", "|---|---|---|"]
    for name, c in contrasts.items():
        lo, hi = c["bootstrap_95_interval"]
        lines.append(f"| {name} | {c['mean']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ["", "## Full validation grid", "",
              "| Arm | Weight | Goal distance | Success | Mean std | Correlation rank |",
              "|---|---|---|---|---|---|"]
    for i, kind in enumerate(KINDS):
        for weight in report["grid"][kind]:
            group = [r for r in report["validation"] if r["kind"] == kind and r["weight"] == weight]
            def mean(key):
                return np.mean([r[key] for r in group])
            lines.append(f"| {LABELS[i]} | {weight} | {mean('goal_distance'):.5f} | "
                         f"{mean('success_rate'):.1%} | {mean('std_mean'):.4f} | {mean('correlation_effective_rank'):.2f} |")
    lines += ["", "Grid-boundary selections: " + (", ".join(boundary) if boundary else "none") + ". "
              "Boundary choices limit claims about adequate tuning. Equal numbers of settings "
              "do not guarantee equally favorable ranges. The combined arm's internal correlation "
              "weight was fixed at 1 and was not separately tuned.", "",
              "## Implementation and scope", "",
              "Floor = mean(ReLU(1-sqrt(sample_variance+1e-4))^2), computed across batch "
              "separately per time. Combined adds mean squared off-diagonal sample correlation, "
              "normalized by D*(D-1), at relative weight 1. No mean anchor or upper variance "
              "bound. This differs from the earlier mean+floor+covariance MomentRegularizer. "
              "With batch 64 and width 192, zero sample correlation across all active dimensions "
              "is unattainable; decorrelation is a soft objective. Exact constant collapse is "
              "stationary for the floor; this tests prevention from random initialization.", "",
              "Fixed 529,790-parameter CNN/JEPA, width 192, existing BatchNorm layers, original "
              "one-step latent prediction, no decoder. AdamW lr 5e-4, decay 1e-3, clipping 1, "
              "batch 64. Same initialization and minibatches per seed. SIGReg uses 64 projections "
              "and 17 frequencies. Train 256 episodes (12000), validation 128 (24000), test 128 "
              "(72000). Five-step CEM with 64 candidates, two iterations, eight elites. All "
              "models use the same 24 test goals. Success threshold 0.12. Simulator state is "
              "used for scoring and diagnostic probes, never as a training target.", "",
              f"Source commit `{report['git_commit']}`; source hashes and per-goal measurements "
              "in results.json. No full published benchmark or cross-dataset replication. "
              "[Frozen protocol](../../VARIANCE_STUDY.md)."]
    (args.directory / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (args.directory / "analysis.json").write_text(json.dumps({"contrasts": contrasts,
            "boundary_selections": boundary}, indent=2)+"\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    metrics = [(distance, "Goal distance (lower better)", False),
               (matrix("std_mean"), "Mean coordinate standard deviation", True),
               (matrix("correlation_effective_rank"), "Correlation effective rank", False),
               (matrix("position_probe_mse"), "Physical position probe MSE (lower better)", True)]
    x = np.arange(4)
    for ax, (values, title, log) in zip(axes.flat, metrics):
        for row in values:
            ax.scatter(x, row, color="#94a3b8", alpha=.7, s=20)
        ax.plot(x, values.mean(0), "o-", color="#2563eb", lw=2)
        ax.set_xticks(x, ["None", "Floor", "Floor + corr.", "SIGReg"])
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 1].axhline(1, ls="--", color="#d97706", label="Floor threshold")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Fresh test: five paired seeds, validation-selected coefficients")
    fig.savefig(args.directory / "variance-comparison.png", dpi=150)
    plt.close(fig)
    print("\n".join(lines[:50]))


if __name__ == "__main__":
    main()
