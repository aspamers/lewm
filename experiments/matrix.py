"""Print a reproducible benchmark command manifest; --run executes it."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def commands(data, seeds, epochs):
    for seed in seeds:
        variants = [("reference", 192, 0, .09, 1)]
        for width in (16, 32):
            variants += [("compact", width, 0, .09, 1),
                         ("compact_low_beta", width, 0, .09, .1),
                         ("predictive", width, 1, 0, .1),
                         ("predictive_sigreg", width, 1, .09, .1)]
        for name, width, obs, sig, beta in variants:
            run = f"{data}-{name}-d{width}-s{seed}"
            cmd = [sys.executable, "train.py", "--config-name=bottleneck",
                   f"data={data}", f"seed={seed}", f"latent_dim={width}",
                   f"subdir={run}", f"trainer.max_epochs={epochs}",
                   f"experiment.observation_weight={obs}",
                   f"experiment.sigreg_weight={sig}", f"experiment.latent_weight={beta}"]
            if name == "reference":
                cmd += ["model=lewm"]
            elif not obs:
                cmd += ["model.decoder=null"]
            yield cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tworoom", choices=["tworoom", "pusht", "dmc", "ogb"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    manifest = list(commands(args.data, args.seeds, args.epochs))
    print(json.dumps(manifest, indent=2))
    if args.run:
        for command in manifest:
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
