"""Summarize a completed sweep and plot mixture-axis histograms as SVG."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"gaussian": "Original SIGReg target", "mixture": "Two-Gaussian mixture",
          "matched_gaussian": "Gaussian, mixture covariance", "moments": "Moment constraints",
          "none": "No regularizer"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "results.json").read_text())
    expected = len(report["seeds"]) * len(LABELS)
    if len(report["test"]) != expected:
        raise ValueError("Wait for the full validation sweep and final test evaluation")
    lines = ["# Direct regularizer comparison", "",
             "**Scope: synthetic pilot, not the published LeWM/TwoRoom benchmark.** "
             "All arms have the same 529,790-parameter model and 192-dimensional latent. "
             "There is no decoder, reconstruction loss, or change in latent-loss weight.", "",
             f"Completed {len(report['validation'])} training runs, each with {report['steps']} updates. "
             "One coefficient per family was chosen by mean validation planning distance across "
             f"{len(report['seeds'])} initialization seeds, then evaluated on separate test trajectories. "
             "The table reports mean ± sample standard deviation across those seeds.", "",
             "| Regularizer | Selected weight | Test goal distance ↓ | Test success ↑ | Position probe MSE ↓ | Latent variance | Effective rank |",
             "|---|---|---|---|---|---|---|"]
    for kind, label in LABELS.items():
        rows = [r for r in report["test"] if r["kind"] == kind]
        def fmt(key, percent=False):
            values = [r[key] * (100 if percent else 1) for r in rows]
            spread = statistics.stdev(values) if len(values) > 1 else 0
            return f"{statistics.mean(values):.4g} ± {spread:.2g}" + ("%" if percent else "")
        lines.append(f"| {label} | {report['selected_weights'][kind]} | {fmt('goal_distance')} | "
                     f"{fmt('success_rate', True)} | {fmt('position_probe_mse')} | {fmt('latent_variance')} | {fmt('effective_rank')} |")
    first = report["test"][0]
    lines += ["", f"Test controls: staying still gives mean goal distance {first['stay_distance']:.4g}; "
              f"a fixed random-action plan gives {first['random_distance']:.4g}.", "",
              "## What changes, and what stays fixed", "",
              "SIGReg's estimator, random unit projections, frequency grid and quadrature weights "
              "are unchanged for the three distribution-target arms. Only the analytical target "
              "characteristic function changes. Tests verify the Gaussian arm exactly equals upstream SIGReg.", "",
              "The mixture is ½ N(+m, 0.25 I) + ½ N(−m, 0.25 I), with each m coordinate √0.75. "
              "Every coordinate has zero mean and unit marginal variance, but covariance is "
              "0.25 I + m mᵀ. The matched Gaussian has that same covariance and a single mode. "
              "This control distinguishes mode structure from changed covariance/goal geometry.", "",
              "A fixed mixture is an alternative prescribed shape, not a superset of the Gaussian "
              "constraint. The moment arm is the genuinely broader distribution family: it anchors "
              "the mean, penalizes standard deviations below one, and penalizes off-diagonal "
              "covariance without prescribing higher moments or the number of modes.", "",
              "CF-target weights: 0.03, 0.09, 0.27. Moment weights: 0.3, 3, 30 because its loss has "
              "a different numerical scale. These are small, predefined searches, not proof of "
              "optimal tuning for any arm. The relative weights inside the moment penalty were "
              "not tuned; finite-batch covariance penalties can trade against its soft variance "
              "floor. The no-regularizer control has weight zero.", "",
              "## Model, data and evaluation", "",
              "A small CNN replaces the ViT. The predictor has two transformer blocks, hidden width "
              "64, four heads, and MLP width 128. Latents and action embeddings are 192-wide; "
              "projectors use BatchNorm like upstream. AdamW uses lr 5e-4, weight decay 1e-3, "
              "batch size 64, gradient clipping 1. All training uses the original four-frame "
              "one-step latent objective. No simulator state enters the training loss.", "",
              "256 training episodes (data seed 12000), 128 validation episodes (24000), "
              "128 final test episodes (36000). Model initialization seeds are 0, 1, 2. "
              "Minibatch schedules are identical across arms within each seed. Probes are fit "
              "on training latents and evaluated on held-out latents; they are diagnostics only.", "",
              "Planning uses 24 fixed nontrivial episodes per split, horizon five, 64 candidates, "
              "two CEM iterations, eight elites, and Euclidean terminal latent distance. The "
              "model selects the plan; simulator state is used only to score its execution. "
              "Success means terminal distance below 0.12. A known-dynamics unit test checks "
              "that the planner improves on random actions and staying still.", "",
              "These are short, open-loop synthetic control problems, not navigation benchmark "
              "success or closed-loop MPC. Seed variation and only 24 test goals limit precision; "
              "the same goals are reused across model seeds, so they are not 72 independent tasks. "
              "Test per-episode distances and all validation outcomes are in results.json.", "",
              "## Does the mixture form modes?", "",
              "The histogram axis is the average latent coordinate, aligned with the fixed mixture's "
              "mean direction. Its prescribed centers are ±√0.75. This axis has no assumed "
              "physical meaning. Balanced positive/negative counts alone do not establish bimodality. "
              "The other arms need not populate this arbitrarily chosen direction.", "",
              "![Test latent distributions along the mixture direction](mixture-axis.png)", "",
              "The full ViT/model configuration is config/train/regularizer_study.yaml. "
              "Published-data training and benchmark evaluation remain outstanding."]
    (args.directory / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True, layout="constrained")
    fig.suptitle("Test latents along the fixed mixture direction", fontsize=16)
    centers = np.linspace(-2, 2, 33)[:-1] + 4/64
    for ax, (kind, label) in zip(axes, LABELS.items()):
        rows = [r for r in report["test"] if r["kind"] == kind]
        hist = [statistics.mean(r["mixture_axis_histogram"][i] / 1024 for r in rows) for i in range(32)]
        ax.bar(centers, hist, width=.12, color="#2563eb")
        for center in (-.75**.5, .75**.5):
            ax.axvline(center, color="#d97706", ls="--", lw=1)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_ylabel("Bin mass")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Mean latent coordinate; dashed lines = prescribed mixture centers")
    axes[-1].set_xlim(-2, 2)
    fig.savefig(args.directory / "mixture-axis.png", dpi=150)
    fig.savefig(args.directory / "mixture-axis.svg")
    svg_path = args.directory / "mixture-axis.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
