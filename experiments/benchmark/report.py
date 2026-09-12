"""Summarize completed screening runs without changing coefficient selection."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    manifest = read(root/"manifest.json")
    if read(root/"status.json")["stage"] != "complete":
        raise RuntimeError("Wait for all declared environments to finish")
    rows = []
    lines = ["# TwoRoom and PushT: bounded real-environment screening", "",
             "Gaussian SIGReg versus variance floor plus decorrelation. These are short",
             "runs of the full LeWM architecture, not the published 100-epoch schedule.", "",
             "## Held-out planning success", "",
             "Selection uses mean validation success across the two model seeds. Fixed",
             "coefficients (SIGReg 0.09, floor + decorrelation 0.3) were transferred",
             "from the synthetic pilot. Test goals were reserved before training.", "",
             "| Environment | Setting | Regularizer | Weight | Seed 31 | Seed 32 | Mean |", 
             "|---|---|---|---:|---:|---:|---:|"]
    controls = []
    for env in manifest["settings"]["environments"]:
        folder = root/env
        tests, validation = read(folder/"test.json"), read(folder/"validation.json")
        for setting in ("selected", "fixed"):
            for kind in manifest["settings"]["grid"]:
                chosen = sorted([x for x in tests if x[setting] and x["kind"] == kind], key=lambda x:x["seed"])
                scores = [x["success_rate"] for x in chosen]
                row = {"environment": env, "setting": setting, "kind": kind,
                       "weight": chosen[0]["weight"], "scores": scores, "mean": statistics.mean(scores)}
                rows.append(row)
                lines.append(f"| {env} | {setting} | {kind} | {row['weight']:g} | " +
                             " | ".join(f"{s:.2f}%" for s in scores) + f" | {row['mean']:.2f}% |")
        control = read(folder/"random-test.json")
        controls.append(f"* {env}: random-policy success {control['success_rate']:.2f}% on the same test goals.")
    lines += ["", *controls, "", "## Training cost", "",
              "Wall time includes data loading, optimization, progress output and checkpoints.",
              "Runs were sequential in a fixed order, so timing is descriptive rather than",
              "a randomized systems benchmark. Peak allocation is PyTorch's allocated memory.", "",
              "| Environment | Regularizer | Median seconds / run | Median peak GiB |", "|---|---|---:|---:|"]
    costs = []
    for env in manifest["settings"]["environments"]:
        validation = read(root/env/"validation.json")
        for kind in manifest["settings"]["grid"]:
            group = [x for x in validation if x["kind"] == kind]
            seconds = statistics.median(x["train_seconds"] for x in group)
            gib = statistics.median(x["peak_allocated_bytes"]/1024**3 for x in group)
            costs.append({"environment": env, "kind": kind, "median_seconds": seconds, "median_peak_gib": gib})
            lines.append(f"| {env} | {kind} | {seconds:.1f} | {gib:.3f} |")
    if (root/"regularizer-cost.json").exists():
        cost = read(root/"regularizer-cost.json")
        lines += ["", "Isolated forward/backward timing uses identical (4, 64, 192) bf16",
                  "embeddings, 40 warmups and 200 alternating-order measurements per arm.", "",
                  "| Regularizer | Median GPU ms | Median wall ms |", "|---|---:|---:|"]
        for kind, values in cost["results"].items():
            lines.append(f"| {kind} | {values['median_gpu_ms']:.3f} | {values['median_wall_ms']:.3f} |")
    if (root/"diagnostics.json").exists():
        lines += ["", "## Embedding spread", "",
                  "Computed from the same 512 reserved test clips for every checkpoint,",
                  "using one frame per clip. Rank must be interpreted alongside absolute spread.", "",
                  "| Environment | Regularizer | Weight | Seed | Mean std | Entropy rank | Mean absolute correlation |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in read(root/"diagnostics.json"):
            lines.append(f"| {row['environment']} | {row['kind']} | {row['weight']:g} | {row['seed']} | "
                         f"{row['std_mean']:.4f} | {row['covariance_entropy_rank']:.2f} | {row['mean_absolute_correlation']:.3f} |")
    lines += ["", "## Interpretation limits", "",
              "* Two model seeds and 16 goals per split provide screening evidence only. Paired",
              "  model seeds reuse goals; 32 outcomes are not 32 independent environments.",
              "* Each run receives 500 updates at batch 64. The CEM budget is reduced to",
              "  64 candidates and three iterations. Poor scores can reflect undertraining",
              "  or insufficient planning search, not an inherent limit of the regularizer.",
              "* Compare learned-policy success with the random control before attributing",
              "  value to either learned representation. Equal success does not prove equivalence.",
              "* Only coefficients transfer from the synthetic pilot; trained model weights",
              "  do not transfer between environments. Cube and Reacher remain untested.",
              "* Original archives omit evaluation seeds. Deterministic reset seeds are",
              "  generated, then recorded start and goal states are restored. Per-environment",
              "  data-check.json records rendered-image and one-step simulator discrepancies.", "",
              "Protocol, source hashes, partitions, validation scores, per-goal test outcomes",
              "and timing are retained beside this report. No test results guide selection.", ""]
    (root/"REPORT.md").write_text("\n".join(lines))
    (root/"summary.json").write_text(json.dumps({"performance": rows, "cost": costs}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
