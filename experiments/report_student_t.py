"""Report the completed Student-t sweep and compare learned/target projections."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"gaussian": "Gaussian / SIGReg", "student_t3": "Student-t, df=3",
          "student_t5": "Student-t, df=5", "student_t9": "Student-t, df=9"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "results.json").read_text())
    if len(report["test"]) != len(LABELS)*len(report["seeds"]):
        raise ValueError("The sweep and independent test evaluation must finish first")
    lines = ["# Student-t versus Gaussian regularization", "",
             "**Synthetic pilot, not a published TwoRoom benchmark.** Fixed 529,790-parameter model, "
             "192-dimensional latent, original one-step latent loss, no decoder. The only training "
             "changes are the distribution target and its coefficient.", "",
             f"{len(report['validation'])} training runs, {report['steps']} updates each. "
             f"Seeds: {report['seeds']}. Coefficients 0.03, 0.09, 0.27 were compared for every target.", "",
             f"Validation selected **{LABELS[report['selected_student_kind']]}** as the primary Student-t "
             "candidate before opening the final test split. Its search has nine df/weight "
             "combinations, versus three weights for Gaussian; the independent test limits "
             "selection optimism but does not make those search budgets equal.", "",
             "Each table entry is mean ± sample standard deviation across model seeds. "
             "The same 24 test goals are reused across seeds, not 72 independent tasks.", "",
             "| Target | Selected weight | Test goal distance ↓ | Test success ↑ | Position probe MSE ↓ | Latent variance | Effective rank |",
             "|---|---|---|---|---|---|---|"]
    for kind, label in LABELS.items():
        rows = [r for r in report["test"] if r["kind"] == kind]
        def fmt(key, pct=False):
            values = [r[key]*(100 if pct else 1) for r in rows]
            sd = statistics.stdev(values) if len(values)>1 else 0
            return f"{statistics.mean(values):.4g} ± {sd:.2g}" + ("%" if pct else "")
        lines.append(f"| {label} | {report['selected_weights'][kind]} | {fmt('goal_distance')} | "
                     f"{fmt('success_rate', True)} | {fmt('position_probe_mse')} | "
                     f"{fmt('latent_variance')} | {fmt('effective_rank')} |")
    first = report["test"][0]
    lines += ["", f"Final test controls: stay-put distance {first['stay_distance']:.4g}; "
              f"random-plan distance {first['random_distance']:.4g}.", "",
              "## Target and implementation", "",
              "Each Student-t is a spherical multivariate distribution: X = G sqrt((ν−2)/U), "
              "where G ~ N(0,I) and U ~ chi-square(ν), with one shared U per vector. "
              "The mean is zero and covariance is identity for all tested ν. This changes "
              "higher-order shape without prescribing the correlated covariance of the previous mixture. "
              "It is not a product of independent univariate t distributions.", "",
              "SIGReg's random directions, 17 frequency points, 64 pilot projections, integration "
              "weights and Gaussian frequency window remain unchanged. Exact Student-t target "
              "characteristic functions are precomputed using exponential-polynomial formulas "
              "for ν=3,5,9. No target samples are drawn during training. Numerical tests compare "
              "these formulas with independent multivariate Student-t samples and verify unit variance.", "",
              "Matching target covariance does not guarantee matching learned covariance: these are "
              "soft finite-projection losses trained for a limited budget. The target latent variance "
              "trace is 192; compare it with actual variance in the table. ν=3 has no finite fourth "
              "moment, so sample kurtosis is not a reliable pass/fail criterion.", "",
              "## Tail diagnostics", "",
              "These plots pool fixed random projections of test embeddings without whitening or "
              "rescaling them. Dashed curves are variance-one target densities. Counts include "
              "correlated projections and adjacent frames; they are visualization data, not "
              "independent statistical trials. The displayed range is limited to [−8,8].", "",
              "![Learned projections and target densities](student-t-tails.png)", "",
              "| Target | Mean projection second moment | Fraction with abs(projection)>3 | Fraction >5 |",
              "|---|---|---|---|"]
    for kind, label in LABELS.items():
        rows = [r for r in report["test"] if r["kind"] == kind]
        def average(key):
            return statistics.mean(r[key] for r in rows)
        lines.append(f"| {label} | {average('projection_second_moment'):.4g} | "
                     f"{average('projection_tail_gt3'):.4%} | {average('projection_tail_gt5'):.4%} |")
    lines += ["", "## Protocol and limits", "",
              "Train: 256 episodes, data seed 12000. Validation: 128 episodes, seed 24000. "
              "Final test: 128 episodes, **fresh seed 48000**, replacing the previously inspected "
              "36000 test split. Results therefore should not be compared numerically to the earlier "
              "mixture table; the Gaussian baseline was rerun here on the same new test set.", "",
              "AdamW lr 5e-4, weight decay 1e-3, batch 64, clipping 1; identical initialization and "
              "minibatches across targets per seed. Small CNN, BatchNorm projectors, two-block "
              "hidden-64 predictor. No simulator state in training. Probes are fit on training "
              "latents for diagnostics only. Coefficients and primary Student-t df are selected "
              "by mean validation planning distance, never final-test performance.", "",
              "All planning uses five-step open-loop CEM with Euclidean terminal latent distance, "
              "64 candidates, two iterations, eight elites and 24 fixed nontrivial goals. "
              "Success is terminal position distance below 0.12. The initial goal distance must "
              "exceed 0.2. Simulator state scores the selected plan but is not provided to the planner.", "",
              "This tests these fixed Student-t targets on short synthetic dynamics, not all "
              "heavy-tailed priors or full LeWM benchmark performance. Tail relaxation is not "
              "a reduction in dense-network computation. Longer runs, more seeds and actual "
              "benchmark datasets remain separate work.", "",
              "Mathematical references: [SciPy multivariate-t parameterization]"
              "(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html) "
              "and [NIST half-integer Bessel formulas](https://dlmf.nist.gov/10.49)."]
    (args.directory / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Fresh test data: learned projections versus variance-one target", fontsize=14)
    x = np.linspace(-8, 8, 501)
    centers = np.linspace(-8, 8, 81)[:-1] + .1
    for ax, (kind, label) in zip(axes.flat, LABELS.items()):
        rows = [r for r in report["test"] if r["kind"] == kind]
        density = np.mean([np.array(r["projection_histogram"])/r["projection_sample_count"]/.2 for r in rows], axis=0)
        ax.plot(centers, np.where(density>0, density, np.nan), color="#2563eb", label="Learned")
        if kind == "gaussian":
            target = np.exp(-x*x/2)/math.sqrt(2*math.pi)
        else:
            df = int(kind.removeprefix("student_t"))
            target = math.gamma((df+1)/2)/(math.gamma(df/2)*math.sqrt(math.pi*(df-2))) * (1+x*x/(df-2))**(-(df+1)/2)
        ax.plot(x, target, "--", color="#d97706", label="Target")
        ax.set_title(label)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(-8, 8)
        ax.set_xlabel("Projection (raw latent units)")
        ax.set_ylabel("Density (log scale)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)
    fig.savefig(args.directory / "student-t-tails.png", dpi=150)
    plt.close(fig)
    print("\n".join(lines[:19]))


if __name__ == "__main__":
    main()
