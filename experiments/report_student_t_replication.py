"""Paired training-seed analysis of the frozen Student-t replication."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KINDS = ["student_t3", "student_t5", "student_t9", "gaussian"]
LABELS = ["df = 3", "df = 5", "df = 9", "Gaussian"]


def matrix(report, metric):
    rows = report["test"]
    seeds = report["seeds"]
    if len(rows) != len(seeds) * 4:
        raise ValueError("Expected all four targets for every seed")
    lookup = {(r["seed"], r["kind"]): r for r in rows}
    if len(lookup) != len(rows) or any(r["weight"] != .09 for r in rows):
        raise ValueError("Duplicate rows or non-frozen weights")
    return np.array([[lookup[s, k][metric] for k in KINDS] for s in seeds])


def interval(values):
    rng = np.random.default_rng(20260912)
    means = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--original", type=Path,
                        default=Path("experiments/results/student-t-study-1000/results.json"))
    args = parser.parse_args()
    report = json.loads((args.directory / "results.json").read_text())
    old = json.loads(args.original.read_text())
    if set(report["seeds"]) & set(old["seeds"]):
        raise ValueError("Replication and original seeds must be disjoint")
    if report["data_seeds"] != old["data_seeds"] or report["steps"] != old["steps"]:
        raise ValueError("Data and training budget must match")
    for r in report["test"] + old["test"]:
        per_goal = np.array(r["per_episode_distance"])
        if len(per_goal) != 24 or not np.isfinite(per_goal).all():
            raise ValueError("Expected 24 finite planning distances per run")
        if not np.isclose(per_goal.mean(), r["goal_distance"], atol=1e-7):
            raise ValueError("Goal-distance aggregate disagrees with raw measurements")
        if not np.isclose((per_goal < .12).mean(), r["success_rate"], atol=1e-7):
            raise ValueError("Success aggregate disagrees with raw measurements")
    distance = matrix(report, "goal_distance")
    success = matrix(report, "success_rate")
    original = matrix(old, "goal_distance")
    contrasts = {
        "df5 minus mean(df3, df9)": distance[:, 1] - (distance[:, 0] + distance[:, 2])/2,
        **{f"{LABELS[i]} minus Gaussian": distance[:, i]-distance[:, 3] for i in range(3)},
        "df5 minus df3": distance[:, 1]-distance[:, 0],
        "df5 minus df9": distance[:, 1]-distance[:, 2],
    }
    summary = {name: {"mean": float(v.mean()), "bootstrap_95_percent_interval": interval(v),
                      "positive_seeds": int((v>0).sum()), "per_seed": v.tolist()}
               for name, v in contrasts.items()}
    dip = summary["df5 minus mean(df3, df9)"]
    low, high = dip["bootstrap_95_percent_interval"]
    conclusion = ("The df5 distance dip reproduced across new training seeds."
                  if low > 0 else "The df5 distance dip reversed across new training seeds."
                  if high < 0 else "The new seeds do not establish a reproducible df5 distance dip.")
    lines = ["# Student-t: ten new training seeds", "", conclusion, "",
             "Fixed coefficient 0.09; four targets; 1,000 updates; seeds 3 through 12. "
             "40 new runs. Same model, data and 24 planning goals as the original sweep. "
             "No hyperparameter or target selection in this replication.", "",
             "## Primary replication results", "",
             "Mean +/- sample standard deviation across the ten new training seeds.", "",
             "| Target | Goal distance (lower better) | Success | Latent variance | Projection second moment |",
             "|---|---|---|---|---|"]
    variance = matrix(report, "latent_variance")
    moment = matrix(report, "projection_second_moment")
    for i, label in enumerate(LABELS):
        lines.append(f"| {label} | {distance[:, i].mean():.5f} +/- {distance[:, i].std(ddof=1):.5f} | "
                     f"{success[:, i].mean():.1%} +/- {success[:, i].std(ddof=1)*100:.1f} pp | "
                     f"{variance[:, i].mean():.1f} | {moment[:, i].mean():.3f} |")
    lines += ["", f"Primary contrast, df5 minus the average of df3 and df9: **{dip['mean']:+.5f}**, "
              f"paired seed bootstrap 95% interval **[{low:+.5f}, {high:+.5f}]**. "
              f"Positive in {dip['positive_seeds']}/{len(distance)} seeds. Positive means df5 is worse.", "",
              "![Distance by target and training seed](df-replication.png)", "",
              "## Paired distance contrasts", "",
              "| Contrast | Mean difference | Bootstrap 95% interval | Positive seeds |",
              "|---|---|---|---|"]
    for name, result in summary.items():
        lo, hi = result["bootstrap_95_percent_interval"]
        lines.append(f"| {name} | {result['mean']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | "
                     f"{result['positive_seeds']}/{len(distance)} |")
    lines += ["", "## Original versus replication", "",
              "The original three seeds motivated this contrast. The pooled 13-seed result "
              "is descriptive; the ten new seeds above are the primary replication.", "",
              "| Target | Original 3: distance / success | New 10: distance / success | Pooled 13: distance / success |",
              "|---|---|---|---|"]
    old_success = matrix(old, "success_rate")
    for i, label in enumerate(LABELS):
        pooled_d = np.concatenate([original[:, i], distance[:, i]])
        pooled_s = np.concatenate([old_success[:, i], success[:, i]])
        lines.append(f"| {label} | {original[:, i].mean():.5f} / {old_success[:, i].mean():.1%} | "
                     f"{distance[:, i].mean():.5f} / {success[:, i].mean():.1%} | "
                     f"{pooled_d.mean():.5f} / {pooled_s.mean():.1%} |")
    lines += ["", "## Scope and uncertainty", "",
              "Intervals use 20,000 paired training-seed bootstrap resamples, RNG seed 20260912. "
              "Whole seeds are resampled, preserving pairing across targets and all goals within "
              "each run. They describe initialization/minibatch variability conditional on this "
              "fixed training data and already-inspected test split (seed 48000). They do not "
              "measure uncertainty across tasks or environments. Ten seeds remain a small sample.", "",
              "Only the df5-versus-neighbors distance contrast is primary. Other intervals are "
              "descriptive and have no multiple-comparison correction. Threshold success is "
              "secondary and can rank targets differently from mean distance. No new claim about "
              "the best df should be selected from these results without another validation.", "",
              "All targets have theoretical covariance I, but learned variance and tail fit can "
              "differ. This experiment does not separate shape from optimization effects. "
              "Synthetic CNN pilot only, not published LeWM/TwoRoom.", "",
              f"Training source commit: `{report['git_commit']}`. Raw per-seed and per-goal "
              "measurements and source hashes are in results.json. Original data: "
              "../student-t-study-1000/results.json."]
    (args.directory / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (args.directory / "analysis.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x = np.arange(4)
    for row in distance:
        axes[0].plot(x, row, "o-", color="#94a3b8", alpha=.5, lw=.8, ms=3)
    axes[0].plot(x, distance.mean(0), "o-", color="#2563eb", lw=2.5, label="New 10 mean")
    axes[0].plot(x, original.mean(0), "s--", color="#d97706", lw=1.5, label="Original 3 mean")
    axes[0].set_xticks(x, LABELS)
    axes[0].set_ylabel("Mean goal distance (lower better)")
    axes[0].set_title("Each gray line is one new training seed")
    axes[0].legend()
    values = contrasts["df5 minus mean(df3, df9)"]
    axes[1].axhline(0, color="#64748b", lw=1)
    axes[1].scatter(report["seeds"], values, color="#2563eb")
    axes[1].axhline(values.mean(), color="#2563eb", label="Mean paired difference")
    axes[1].axhspan(low, high, alpha=.12, color="#2563eb", label="95% seed bootstrap interval")
    axes[1].set_xlabel("New training seed")
    axes[1].set_ylabel("df5 minus average(df3, df9) distance")
    axes[1].set_title("Positive means df5 performs worse")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.directory / "df-replication.png", dpi=160)
    plt.close(fig)
    print(conclusion)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
