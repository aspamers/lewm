"""Reevaluate frozen saved checkpoints with training-only encoder BN calibration."""
import json
from pathlib import Path
import shutil
import statistics
from types import SimpleNamespace

import torch
from run import ENVIRONMENTS, evaluate, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(4)
    root = Path("outputs/calibrated-reevaluation")
    root.mkdir(exist_ok=True)
    original = Path("outputs/two-environment-benchmark")
    options = SimpleNamespace(candidates=64,cem_steps=3)
    jobs = []
    for env in ENVIRONMENTS:
        for row in read(original/env/"test.json"):
            if row["selected"]:
                checkpoint = original/env/f"{row['kind']}-w{row['weight']}-s{row['seed']}"
                jobs.append({"cohort":"original_500","environment":env,"kind":row["kind"],
                             "weight":row["weight"],"seed":row["seed"],"checkpoint":str(checkpoint)})
    longer = Path("outputs/two-environment-1000/tworoom")
    for row in read(longer/"test.json"):
        jobs.append({"cohort":"original_1000","environment":"tworoom","kind":row["kind"],
                     "weight":row["weight"],"seed":row["seed"],
                     "checkpoint":str(longer/f"{row['kind']}-w{row['weight']}-s{row['seed']}")})
    for out in sorted(Path("outputs/floor-strength").glob("floor*-s31")):
        if (out/"result.json").exists():
            row = read(out/"result.json")
            jobs.append({"cohort":"corrected_fp32_250","environment":"tworoom","kind":"variance_decorrelation",
                         "weight":row["floor_weight"],"correlation_weight":.1,"seed":31,"checkpoint":str(out)})
    write_json(root/"jobs.json",jobs)
    results = []
    for job in jobs:
        checkpoint = Path(job["checkpoint"])
        env = job["environment"]
        partition = read(original/env/"partition.json")
        path = Path("/data/datasets")/ENVIRONMENTS[env][0]
        write_json(root/"status.json",{"stage":"evaluating","completed":len(results),"total":len(jobs),"job":job})
        before = evaluate(path,partition,env,checkpoint,"test",options,calibrate=False)
        after = evaluate(path,partition,env,checkpoint,"test",options,calibrate=True)
        results.append({**job,"before":before,"after":after})
        write_json(root/"results.json",results)
        print(job, before["success_rate"], "->", after["success_rate"], flush=True)
    lines = ["# Planning reevaluation after training-only encoder BN calibration", "",
             "No model weights were retrained or changed. Only the encoder projection BatchNorm",
             "running mean/variance were recalculated using the same 512 training clips per environment.",
             "Predictor normalization remains unchanged. The original selected coefficients remain frozen.",
             "Planning uses the same 16 goals, seeds and reduced CEM budget as before.", "",
             "## Individual results", "",
             "| Cohort | Environment | Regularizer | Weight | Seed | Original success | Calibrated success |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['cohort']} | {row['environment']} | {row['kind']} | {row['weight']:g} | {row['seed']} | "
                     f"{row['before']['success_rate']:.2f}% | {row['after']['success_rate']:.2f}% |")
    lines += ["", "## Means within matched cohorts", "",
              "| Cohort | Environment | Regularizer | Weight | Seeds | Original mean | Calibrated mean |",
              "|---|---|---|---:|---:|---:|---:|"]
    keys = sorted({(r["cohort"],r["environment"],r["kind"],r["weight"]) for r in results})
    for cohort,env,kind,weight in keys:
        rows = [r for r in results if (r["cohort"],r["environment"],r["kind"],r["weight"])==(cohort,env,kind,weight)]
        lines.append(f"| {cohort} | {env} | {kind} | {weight:g} | {len(rows)} | "
                     f"{statistics.mean(r['before']['success_rate'] for r in rows):.3f}% | "
                     f"{statistics.mean(r['after']['success_rate'] for r in rows):.3f}% |")
    lines += ["", "Random-policy scores on these same goals: TwoRoom 37.5%, PushT 0%.", "",
              "Original 500/1,000-update cohorts used the old loss implementation and mixed precision.",
              "The corrected 250-update cohort uses full precision and independent floor coefficients",
              "with correlation coefficient 0.1. It currently has only seed 31 and is not a matched",
              "comparison with the original Gaussian cohort. Its uncalibrated planning scores were",
              "first measured during this reevaluation. The interrupted sweeps were not resumed.", "",
              "These are exploratory reevaluations of repeatedly inspected goals. Calibration itself",
              "uses training data only, but the broader diagnostic choices followed earlier outcomes.",
              "Fresh independent confirmation is required for general performance claims.", ""]
    (root/"REPORT.md").write_text("\n".join(lines))
    destination = Path("experiments/results/calibrated-reevaluation")
    destination.mkdir(parents=True,exist_ok=True)
    for name in ("REPORT.md","results.json","jobs.json"):
        shutil.copy2(root/name,destination/name)
    write_json(root/"status.json",{"stage":"complete","runs":len(results)})


if __name__=="__main__":
    main()
