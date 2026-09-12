"""Matched retuning followed by frozen-parameter, fresh-seed confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from types import SimpleNamespace

import h5py
import torch
from run import ENVIRONMENTS, train, evaluate, random_control, write_json
from fresh_protocol import CONFIGS, TUNING_SEEDS, CONFIRMATION_SEEDS, fresh_partition


def read(path):
    return json.loads(path.read_text())


def summarize(output, settings, selected, tuning, confirmation):
    lines = ["# Fresh tuning and independent confirmation", "",
             f"Both environments and methods use {settings['steps']} updates, batch 64, float32 training,",
             "encoder activation checkpointing, TF32 disabled, and training-only encoder BN calibration.",
             "Four settings per method and two paired tuning seeds receive equal search budgets.",
             "Coefficients were frozen using 32 validation goals before any confirmation goals were opened.",
             "Three separate confirmation seeds use 64 previously uninspected goal episodes per environment.", "",
             "## Tuning results", "",
             "| Environment | Setting | Seed 1101 | Seed 1102 | Mean | Selected |",
             "|---|---|---:|---:|---:|---|"]
    for env in ENVIRONMENTS:
        for config in CONFIGS:
            group = sorted([r for r in tuning if r["environment"]==env and r["config"]==config],key=lambda r:r["seed"])
            scores = [r["evaluation"]["success_rate"] for r in group]
            chosen = selected[env][config["kind"]]["name"] == config["name"]
            lines.append(f"| {env} | {config['name']} | " + " | ".join(f"{s:.2f}%" for s in scores) +
                         f" | {statistics.mean(scores):.3f}% | {'yes' if chosen else ''} |")
    lines += ["", "## Confirmation results", "",
              "| Environment | Setting | Seed 2101 | Seed 2102 | Seed 2103 | Mean | Random |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for env in ENVIRONMENTS:
        random_score = read(output/env/"random-test.json")["success_rate"]
        for kind,config in selected[env].items():
            scores = [r["evaluation"]["success_rate"] for r in sorted(confirmation,key=lambda r:r["seed"])
                      if r["environment"]==env and r["config"]["kind"]==kind]
            lines.append(f"| {env} | {config['name']} | " + " | ".join(f"{s:.2f}%" for s in scores) +
                         f" | {statistics.mean(scores):.3f}% | {random_score:.2f}% |")
    lines += ["", "## Limits", "",
              "This is a bounded coarse search, not proof of globally optimal coefficients. Ties use",
              "the predefined configuration order. Training and calibration data are unchanged across methods.",
              "Confirmation episodes were excluded from earlier test goals and the historical 512-clip",
              "test-embedding probe. Goal coordinates are chosen from metadata before training; images",
              "and outcomes are first used during final evaluation. All confirmation models train before",
              "any final test evaluation starts. Three seeds share the same goals; their outcomes are",
              "not independent draws of all possible tasks. New episodes remain from the same datasets,",
              "not a new simulator or dataset distribution. No performance-driven stopping or seed selection.", ""]
    (output/"REPORT.md").write_text("\n".join(lines))


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    configs = [CONFIGS[0],CONFIGS[4]] if args.smoke else CONFIGS
    tuning_seeds = [TUNING_SEEDS[0]] if args.smoke else TUNING_SEEDS
    steps = 2 if args.smoke else args.steps
    settings = {"steps":steps,"batch":64,"precision":"float32","configs":configs,
                "tuning_seeds":tuning_seeds,"confirmation_seeds":CONFIRMATION_SEEDS,
                "validation_goals":32,"confirmation_goals":64,"candidates":64,"cem_steps":3,
                "tf32":False,"encoder_gradient_checkpointing":True,"smoke":args.smoke,
                "output":str(args.output),"torch":torch.__version__,"gpu":torch.cuda.get_device_name()}
    files = [Path(__file__),Path("experiments/benchmark/fresh_protocol.py"),Path("experiments/benchmark/run.py"),
             Path("experiments/benchmark/runtime.py"),Path("experiments/benchmark/calibration.py"),
             *Path("src/lewm").glob("*.py"),Path("config/train/model/lewm.yaml"),
             *[Path(f"config/eval/{env}.yaml") for env in ENVIRONMENTS]]
    manifest = {"settings":settings,"sources":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    if (args.output/"manifest.json").exists() and read(args.output/"manifest.json")!=manifest:
        raise RuntimeError("Protocol/source changed; use a fresh output directory")
    write_json(args.output/"manifest.json",manifest)
    partitions = {}
    for env,(filename,_) in ENVIRONMENTS.items():
        folder = args.output/env
        folder.mkdir(exist_ok=True)
        with h5py.File(Path("/data/datasets")/filename,"r") as data:
            partition = fresh_partition(read(Path("outputs/two-environment-benchmark")/env/"partition.json"),data["ep_len"][:])
        if (folder/"partition.json").exists() and read(folder/"partition.json")!=partition:
            raise RuntimeError("Partition changed")
        write_json(folder/"partition.json",partition)
        partitions[env] = partition
    tuning = []
    def fit(env,config,seed,stage):
        checkpoint = args.output/env/stage/f"{config['name']}-s{seed}"
        options = SimpleNamespace(steps=steps,batch=64,precision="float32",candidates=64,cem_steps=3,
                                  **{k:config[k] for k in ("floor_weight","correlation_weight") if k in config})
        path = Path("/data/datasets")/ENVIRONMENTS[env][0]
        write_json(args.output/"status.json",{"stage":stage,"environment":env,"config":config,"seed":seed,
                                             "checkpoint":str(checkpoint)})
        training = train(path.with_name(path.stem+"_frames.h5"),partitions[env],config["kind"],config["weight"],seed,options,checkpoint)
        return checkpoint,options,training
    for env in ENVIRONMENTS:
        path = Path("/data/datasets")/ENVIRONMENTS[env][0]
        for seed in tuning_seeds:
            for config in configs:
                checkpoint,options,training = fit(env,config,seed,"tuning")
                metrics = evaluate(path,partitions[env],env,checkpoint,"validation",options)
                tuning.append({"environment":env,"config":config,"seed":seed,"training":training,"evaluation":metrics})
                write_json(args.output/"tuning.json",tuning)
    if args.smoke:
        write_json(args.output/"status.json",{"stage":"smoke_complete","final_test_opened":False})
        return
    selected = {}
    for env in ENVIRONMENTS:
        selected[env] = {}
        for kind in ("gaussian","variance_decorrelation"):
            eligible = [c for c in configs if c["kind"]==kind]
            chosen = max(eligible,key=lambda c:statistics.mean(r["evaluation"]["success_rate"] for r in tuning
                           if r["environment"]==env and r["config"]==c))
            selected[env][kind] = chosen
    if (args.output/"selected.json").exists() and read(args.output/"selected.json")!=selected:
        raise RuntimeError("Frozen selection changed")
    write_json(args.output/"selected.json",selected)
    fitted = []
    for env in ENVIRONMENTS:
        for seed in CONFIRMATION_SEEDS:
            for config in selected[env].values():
                checkpoint,options,training = fit(env,config,seed,"confirmation_training")
                fitted.append((env,config,seed,checkpoint,options,training))
    confirmation = []
    # No confirmation outcomes are opened until all coefficients and all
    # confirmation model training runs are complete.
    for env in ENVIRONMENTS:
        random_control(Path("/data/datasets")/ENVIRONMENTS[env][0],partitions[env],env,args.output/env,options)
    for env,config,seed,checkpoint,options,training in fitted:
        write_json(args.output/"status.json",{"stage":"confirmation_evaluation","environment":env,"config":config,"seed":seed})
        metrics = evaluate(Path("/data/datasets")/ENVIRONMENTS[env][0],partitions[env],env,checkpoint,"test",options)
        confirmation.append({"environment":env,"config":config,"seed":seed,"training":training,"evaluation":metrics})
        write_json(args.output/"confirmation.json",confirmation)
    summarize(args.output,settings,selected,tuning,confirmation)
    destination = Path("experiments/results")/args.output.name
    destination.mkdir(parents=True,exist_ok=True)
    for name in ("REPORT.md","manifest.json","tuning.json","selected.json","confirmation.json"):
        shutil.copy2(args.output/name,destination/name)
    for env in ENVIRONMENTS:
        (destination/env).mkdir(exist_ok=True)
        for name in ("partition.json","random-test.json"):
            shutil.copy2(args.output/env/name,destination/env/name)
    write_json(args.output/"status.json",{"stage":"complete"})


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--output",type=Path,default=Path("outputs/fresh-sweep"))
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/"status.json",{"stage":"error","error":repr(error)})
        raise
