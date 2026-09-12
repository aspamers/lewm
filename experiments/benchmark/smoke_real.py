"""Short real-data training and two-goal planning integration check."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from run import prepare_data, train, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-real-smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    path = Path("/data/datasets/tworoom.h5")
    partition = prepare_data(path, args.output)
    partition["validation_goals"] = {k: v[:2] for k, v in partition["validation_goals"].items()}
    settings = SimpleNamespace(steps=20, batch=64, candidates=64, cem_steps=3)
    results = {}
    for kind, weight in [("gaussian", .09), ("variance_decorrelation", .3)]:
        out = args.output/kind
        training_path = path.with_name(path.stem+"_frames.h5")
        training = train(training_path if training_path.exists() else path, partition, kind, weight, 997, settings, out)
        evaluation = evaluate(path, partition, "tworoom", out, "validation", settings)
        results[kind] = {"training": training, "evaluation": evaluation}
    (args.output/"results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
