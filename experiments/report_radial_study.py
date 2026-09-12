"""Analyze performance and whether radial target differences were learned."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gamma
import torch

from lewm.regularizers import RADIAL_VARIANCES, make_regularizer
from radial_study import KINDS

LABELS = ["Broad (8x)", "Gaussian", "Half variance", "Quarter variance", "Sphere"]
VARIANCES = [8., 1., .5, .25, 0.]


def bootstrap(values):
    rng = np.random.default_rng(20260912)
    means = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "results.json").read_text())
    seeds, d = report["seeds"], report["width"]
    expected = len(seeds)*sum(len(w) for w in report["grid"].values())
    if len(report["validation"]) != expected or len(report["test"]) != len(seeds)*5:
        raise ValueError("Complete training and final evaluation are required")
    rows = {(r["seed"], r["kind"]): r for r in report["test"]}
    if len(rows) != len(report["test"]):
        raise ValueError("Duplicate test rows")
    for row in report["test"]:
        distances = np.array(row["per_episode_distance"])
        assert len(distances) == 24 and np.isfinite(distances).all()
        assert np.isclose(distances.mean(), row["goal_distance"], atol=1e-7)
        assert np.isclose((distances < .12).mean(), row["success_rate"], atol=1e-7)
        assert row["weight"] == report["selected_weights"][row["kind"]]
    def matrix(metric):
        return np.array([[rows[s, k][metric] for k in KINDS] for s in seeds])
    distance, success = matrix("goal_distance"), matrix("success_rate")
    cv2 = matrix("radius_squared_cv2")
    scale = matrix("radius_squared_mean_over_dim")
    target_cv2 = 2*np.array(VARIANCES)/d
    primary = report["selected_primary_kind"]
    primary_index = KINDS.index(primary)
    contrasts = {}
    for i, kind in enumerate(KINDS):
        diff = distance[:, i]-distance[:, 1]
        contrasts[kind] = {"mean_distance_difference": float(diff.mean()),
                           "paired_bootstrap_95_interval": bootstrap(diff).tolist(),
                           "per_seed_difference": diff.tolist()}
    # Target-only separation is computed with exactly the pilot frequency kernel.
    reference = make_regularizer("gaussian", latent_dim=d)
    directions = torch.zeros(d, 1)
    directions[0] = 1
    separation = {}
    for kind in KINDS:
        reg = make_regularizer(kind, latent_dim=d)
        delta = reg.target_cf(directions)[0] - reference.phi
        separation[kind] = {"max_abs_cf_difference": float(delta.abs().max()),
                            "population_kernel_distance": float(delta.square() @ reference.weights),
                            "batch64_scaled_population_distance": float(64*(delta.square() @ reference.weights)),
                            "cf": reg.target_cf(directions)[0].tolist()}
    primary_diff = contrasts[primary]
    lo, hi = primary_diff["paired_bootstrap_95_interval"]
    # Conclusions deliberately remain descriptive; five-seed intervals do not
    # establish an optimum, and projected loss need not enforce the target radius.
    performance = ("Validation selected Gaussian as the primary target."
                   if primary == "gaussian" else
                   f"Validation selected {LABELS[primary_index]} as the primary target; its fresh-test "
                   f"distance difference from Gaussian is {primary_diff['mean_distance_difference']:+.5f} "
                   f"(paired 95% interval [{lo:+.5f}, {hi:+.5f}]).")
    lines = ["# Radial shape sweep", "", "**Synthetic pilot, not published LeWM/TwoRoom.**", "",
             performance, "", f"{expected} training runs; {report['steps']} updates; seeds {seeds}. "
             "Five targets each received the same coefficient grid 0.03, 0.09, 0.27. "
             "Coefficients and the primary target were chosen by validation planning distance "
             "before evaluating fresh test data seed 60000.", "",
             "## Selected-coefficient test results", "",
             "Mean +/- sample standard deviation across training seeds. Lower distance is better.", "",
             "| Target | Weight | Goal distance | Success | Effective rank |",
             "|---|---|---|---|---|"]
    for i, kind in enumerate(KINDS):
        lines.append(f"| {LABELS[i]} | {report['selected_weights'][kind]} | "
                     f"{distance[:, i].mean():.5f} +/- {distance[:, i].std(ddof=1):.5f} | "
                     f"{success[:, i].mean():.1%} +/- {success[:, i].std(ddof=1)*100:.1f} pp | "
                     f"{matrix('effective_rank')[:, i].mean():.2f} |")
    lines += ["", "![Performance and radial shape](radial-comparison.png)", "",
              "## Did training produce the target shapes?", "",
              "Squared-radius CV^2 = Var(||z||^2) / E[||z||^2]^2. This removes overall scale "
              "without whitening or changing coordinates. Raw radius is about the prescribed zero "
              "origin; centered CV^2 is also reported. A perfect spherical target has CV^2=0. "
              "Mean squared radius / D should be 1 for every target.", "",
              "| Target | Target CV^2 | Learned CV^2 | Centered CV^2 | Mean squared radius / D |",
              "|---|---|---|---|---|"]
    for i in range(5):
        lines.append(f"| {LABELS[i]} | {target_cv2[i]:.6f} | {cv2[:, i].mean():.6f} | "
                     f"{matrix('centered_radius_squared_cv2')[:, i].mean():.6f} | {scale[:, i].mean():.4f} |")
    lines += ["", "Selected coefficients can differ, so the table above does not isolate the "
              "effect of shape at fixed loss strength. The following comparison uses validation "
              "embeddings at the common coefficient 0.09 for every target (not additional "
              "test-set selection).", "",
              "| Target | Target CV^2 | Validation learned CV^2 at weight 0.09 |",
              "|---|---|---|"]
    matched_cv = []
    for i, kind in enumerate(KINDS):
        values = [r['radius_squared_cv2'] for r in report['validation']
                  if r['kind'] == kind and r['weight'] == .09]
        matched_cv.append(float(np.mean(values)))
        lines.append(f"| {LABELS[i]} | {target_cv2[i]:.6f} | {matched_cv[-1]:.6f} |")
    lines += ["", f"At this common weight, sphere versus Gaussian changes mean learned CV^2 by "
              f"{(matched_cv[-1]/matched_cv[1]-1)*100:+.2f}%. The sphere target is zero. "
              "Compare the gap to the target before interpreting a performance tie as evidence "
              "that different exact shapes are equally good."]
    lines += ["", "![Learned radial distributions and target densities](radial-distributions.png)", "",
              "Histograms use raw ||z||/sqrt(D), pooled across models and frames for visualization "
              "only. The sphere's target is a point mass at 1. Histogram range is [0,3]; raw "
              "moments and quantiles include all observations.", "",
              "## Target distinguishability under the current regularizer", "",
              "All targets are isotropic with covariance I. Squared radius follows "
              "Gamma(D/(2v), scale=2v), except v=0 (fixed radius). The target-only kernel "
              "distance integrates squared characteristic-function differences from Gaussian "
              "using the unchanged 17-point SIGReg grid and weights. This is a population "
              "separation diagnostic, not the noisy minibatch loss or a significance test.", "",
              "| Target | Max projected CF difference from Gaussian | Kernel distance | Scaled by batch 64 |",
              "|---|---|---|---|"]
    for i, kind in enumerate(KINDS):
        s = separation[kind]
        lines.append(f"| {LABELS[i]} | {s['max_abs_cf_difference']:.6g} | "
                     f"{s['population_kernel_distance']:.6g} | {s['batch64_scaled_population_distance']:.6g} |")
    lines += ["", "## Paired test distance differences from Gaussian", "",
              "Negative favors the alternative. 20,000 paired seed bootstrap resamples, RNG "
              "20260912. Intervals condition on these data and selected coefficients; they "
              "exclude tuning uncertainty. Five seeds are a small sample. Comparisons other "
              "than the validation-selected primary target are exploratory, with no "
              "multiple-comparison correction.", "",
              "| Target | Mean difference | 95% seed bootstrap interval |", "|---|---|---|"]
    for i, kind in enumerate(KINDS):
        c = contrasts[kind]
        a, b = c["paired_bootstrap_95_interval"]
        lines.append(f"| {LABELS[i]} | {c['mean_distance_difference']:+.5f} | [{a:+.5f}, {b:+.5f}] |")
    lines += ["", "## Full validation grid", "",
              "| Target | Weight | Mean distance | Mean success | Mean radial CV^2 |", "|---|---|---|---|---|"]
    for i, kind in enumerate(KINDS):
        for weight in report["grid"][kind]:
            group = [r for r in report["validation"] if r["kind"] == kind and r["weight"] == weight]
            lines.append(f"| {LABELS[i]} | {weight} | {np.mean([r['goal_distance'] for r in group]):.5f} | "
                         f"{np.mean([r['success_rate'] for r in group]):.1%} | "
                         f"{np.mean([r['radius_squared_cv2'] for r in group]):.6f} |")
    lines += ["", "## Protocol and limitations", "",
              "Fixed 529,790-parameter CNN/JEPA, latent width 192, original one-step loss, no "
              "decoder and no explicit radius loss. Same initialization and minibatches across "
              "targets per seed. AdamW lr 5e-4, decay 1e-3, batch 64, gradient clipping 1. "
              "Train 256 episodes (seed 12000), validation 128 (24000), fresh test 128 (60000). "
              "Five-step CEM, 64 candidates, two rounds, eight elites, same 24 goals for every "
              "model. Success threshold 0.12. Simulator state scores plans, never trains the model.", "",
              "Each target has equal coefficient budget, but selecting among four alternatives "
              "has greater total search budget than Gaussian alone. Test rankings do not choose "
              "a new winner. Generalization across datasets, longer training, and the full LeWM "
              "benchmark remain untested. A target tying or winning while its learned radial "
              "distribution fails to approach that target does not identify an optimal shape.", "",
              f"Training source commit: `{report['git_commit']}`. SciPy {scipy.__version__} used for "
              "target construction and this report. Raw per-goal metrics, histograms, quantiles "
              "and source hashes are in results.json; derived contrasts and target CFs are in analysis.json.", "",
              "[Protocol and derivation](../../RADIAL_STUDY.md). Mathematical references: "
              "[NIST 1F1 series](https://dlmf.nist.gov/13.2) and "
              "[SciPy 0F1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp0f1.html)."]
    (args.directory / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    analysis = {"primary_kind": primary, "contrasts": contrasts, "target_separation": separation,
                "target_cv2": dict(zip(KINDS, target_cv2.tolist())),
                "learned_cv2": dict(zip(KINDS, cv2.mean(0).tolist())), "scipy": scipy.__version__}
    (args.directory / "analysis.json").write_text(json.dumps(analysis, indent=2)+"\n")
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for row in distance:
        axes[0].plot(x, row, "o-", color="#94a3b8", alpha=.45, lw=.8, ms=3)
    axes[0].plot(x, distance.mean(0), "o-", color="#2563eb", lw=2, label="Mean of five seeds")
    axes[0].set_ylabel("Mean goal distance (lower better)")
    axes[0].set_title("Fresh test; validation-selected coefficients")
    axes[0].legend()
    matched_matrix = np.array([[next(r['radius_squared_cv2'] for r in report['validation']
                                    if r['seed'] == s and r['kind'] == k and r['weight'] == .09)
                                for k in KINDS] for s in seeds])
    for row in matched_matrix:
        axes[1].scatter(x, row, color="#94a3b8", alpha=.6, s=15)
    axes[1].plot(x, matched_matrix.mean(0), "o-", color="#2563eb", label="Learned mean")
    axes[1].plot(x, target_cv2, "s--", color="#d97706", label="Target")
    axes[1].set_ylabel("Squared-radius CV squared")
    axes[1].set_title("Validation radius at common weight 0.09")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, ["Broad", "Gaussian", "Half", "Quarter", "Sphere"])
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.directory / "radial-comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharex=True, sharey=True, layout="constrained")
    r = np.linspace(.001, 2, 1200)
    centers = np.arange(60)*.05+.025
    for i, (ax, kind, v) in enumerate(zip(axes, KINDS, VARIANCES)):
        density = np.mean([np.array(rows[s, kind]["radius_over_sqrt_dim_histogram"])/
                           rows[s, kind]["radius_sample_count"]/.05 for s in seeds], axis=0)
        ax.plot(centers, density, color="#2563eb", label="Learned")
        if v:
            k = d/(2*v)
            ax.plot(r, gamma.pdf(r*r, a=k, scale=1/k)*2*r, "--", color="#d97706", label="Target")
        else:
            ax.axvline(1, ls="--", color="#d97706", label="Target point mass")
        ax.set_title(f"{LABELS[i]}\nweight {report['selected_weights'][kind]}")
        ax.set_xlim(.4, 1.7)
        ax.set_xlabel("Radius / sqrt(D)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8)
    fig.savefig(args.directory / "radial-distributions.png", dpi=150)
    plt.close(fig)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
