"""Full LeWM commands for the direct regularizer study. --run executes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from regularizer_study import GRID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tworoom", choices=["tworoom", "pusht", "dmc", "ogb"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    commands = []
    for seed in args.seeds:
        for kind, weights in GRID.items():
            for weight in weights:
                name = f"{args.data}-{kind}-w{weight}-s{seed}"
                commands.append([sys.executable, "train.py", "--config-name=regularizer_study",
                                 f"data={args.data}", f"target={kind}", f"seed={seed}",
                                 f"experiment.sigreg_weight={weight}", f"subdir={name}",
                                 f"trainer.max_epochs={args.epochs}"])
    print(json.dumps(commands, indent=2))
    if args.run:
        for command in commands:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == "__main__":
    main()
