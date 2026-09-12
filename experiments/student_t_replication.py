"""Repeat the df comparison with ten new training seeds and frozen settings."""
import argparse
from pathlib import Path

from regularizer_study import run
from student_t_study import tail_diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(3, 13)))
    parser.add_argument("--output", type=Path, default=Path("outputs/student-t-replication-1000"))
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("steps must be positive and seeds must be unique")
    if (args.output / "results.json").exists():
        parser.error("output already contains results; choose a new directory")
    run(args, grid={kind: [.09] for kind in
                   ("gaussian", "student_t3", "student_t5", "student_t9")},
        test_seed=48000, extra_diagnostics=tail_diagnostics, fixed_comparison=True,
        sources=[Path(__file__), Path("experiments/student_t_study.py")])
