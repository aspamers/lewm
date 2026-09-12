"""Fresh 1,000-update reruns with coefficients frozen by the 500-update screen."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from types import SimpleNamespace

import torch

from run import ENVIRONMENTS, train, evaluate, write_json
from diagnostics import diagnose


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/two-environment-benchmark"))
    parser.add_argument("--output", type=Path, default=Path("outputs/two-environment-1000"))
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    old = read(args.source/"manifest.json")
    if read(args.source/"status.json")["stage"] != "complete":
        raise RuntimeError("The source screen must be complete")
    selected = {env: read(args.source/env/"selected.json") for env in old["settings"]["environments"]}
    settings = {**old["settings"], "output": str(args.output), "steps": old["settings"]["steps"]*2,
                "selected": selected, "selection_source": str(args.source),
                "restart": "from_scratch", "precision": "float32",
                "evaluation": "same goals; exploratory budget comparison"}
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [*Path("experiments/benchmark").glob("*.py"),
                         *Path("src/lewm").glob("*.py"), Path("config/train/model/lewm.yaml"),
                         Path("config/eval/tworoom.yaml"), Path("config/eval/pusht.yaml")]}
    manifest = {"settings": settings, "sources": sources, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(), "source_manifest": old}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"manifest.json").exists() and read(args.output/"manifest.json") != manifest:
        raise RuntimeError("Cannot mix settings or source versions in an existing run")
    write_json(args.output/"manifest.json", manifest)
    options = SimpleNamespace(**settings)
    for env in settings["environments"]:
        out = args.output/env
        out.mkdir(exist_ok=True)
        for filename in ("partition.json", "data-check.json", "random-test.json", "selected.json"):
            shutil.copy2(args.source/env/filename, out/filename)
        partition = read(out/"partition.json")
        path = args.data_root/"datasets"/ENVIRONMENTS[env][0]
        cache = path.with_name(path.stem+"_frames.h5")
        validation, tests = [], []
        for seed in settings["seeds"]:
            for kind, weight in selected[env].items():
                checkpoint = out/f"{kind}-w{weight}-s{seed}"
                write_json(args.output/"status.json", {"stage": "training", "run": str(checkpoint)})
                training = train(cache, partition, kind, weight, seed, options, checkpoint)
                write_json(args.output/"status.json", {"stage": "evaluation", "run": str(checkpoint)})
                val = evaluate(path, partition, env, checkpoint, "validation", options)
                validation.append({**training, **val})
                write_json(out/"validation.json", validation)
                metrics = evaluate(path, partition, env, checkpoint, "test", options)
                tests.append({**training, "selected": True, "fixed": False, **metrics})
                write_json(out/"test.json", tests)
    write_json(args.output/"status.json", {"stage": "complete"})
    write_json(args.output/"finalization.json", {"stage": "diagnostics"})
    diagnose(args.output, args.data_root)
    lines = ["# Selected-coefficient reruns at double the training budget", "",
             f"Eight fresh runs, {settings['steps']} updates each, batch {settings['batch']}, seeds {settings['seeds']}.",
             "Coefficients are frozen from the earlier validation selection; no new coefficient search.",
             "Architecture, learning rate, prediction objective, partitions and planning budget are unchanged.",
             "This executable now uses the corrected correlation loss and float32 training. Against the archived original",
             "mixed-precision screen this changes methodology as well as training duration, so it is not a pure budget ablation.",
             "The same test goals are reused to compare budgets. This is exploratory follow-up, not independent confirmation.",
             "Fresh runs share initialization seeds and paired minibatches across methods. The longer sampler draws a new clip",
             "sequence, so these are not exact continuations of the earlier 500-update trajectories.", "",
             "| Environment | Regularizer | Weight | 500-update mean | 1,000-update seeds | 1,000-update mean | Random |",
             "|---|---|---:|---:|---|---:|---:|"]
    for env in settings["environments"]:
        before, after = read(args.source/env/"test.json"), read(args.output/env/"test.json")
        for kind, weight in selected[env].items():
            previous = [r["success_rate"] for r in before if r["selected"] and r["kind"] == kind]
            current = [r["success_rate"] for r in sorted(after, key=lambda r:r["seed"]) if r["kind"] == kind]
            random_score = read(args.output/env/"random-test.json")["success_rate"]
            lines.append(f"| {env} | {kind} | {weight:g} | {statistics.mean(previous):.3f}% | " +
                         ", ".join(f"{v:.2f}%" for v in current) + f" | {statistics.mean(current):.3f}% | {random_score:.2f}% |")
    lines += ["", "## Evaluation embedding spread", "",
              "| Environment | Regularizer | Seed | Mean coordinate std | Entropy rank |",
              "|---|---|---:|---:|---:|"]
    for r in read(args.output/"diagnostics.json"):
        lines.append(f"| {r['environment']} | {r['kind']} | {r['seed']} | {r['std_mean']:.5f} | {r['covariance_entropy_rank']:.2f} |")
    lines += ["", "Two seeds and 16 shared goals remain a small sample. Relative rank needs absolute spread",
              "to interpret collapse; inference-mode measurements can also depend on BatchNorm running statistics.", ""]
    (args.output/"REPORT.md").write_text("\n".join(lines))
    destination = Path("experiments/results/two-environment-1000")
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("REPORT.md", "manifest.json", "diagnostics.json"):
        shutil.copy2(args.output/filename, destination/filename)
    for env in settings["environments"]:
        (destination/env).mkdir(exist_ok=True)
        for filename in ("test.json", "validation.json", "selected.json", "random-test.json", "partition.json"):
            shutil.copy2(args.output/env/filename, destination/env/filename)
    write_json(args.output/"finalization.json", {"stage": "complete", "report": str(destination/"REPORT.md")})
    print("DOUBLE-BUDGET BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
