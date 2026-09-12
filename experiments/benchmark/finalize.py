"""Finish diagnostics and publish a compact local result bundle after the sweep."""
import argparse
import json
from pathlib import Path
import shutil

from diagnostics import diagnose
from cost import measure
from report import summarize
from run import write_json


def finalize(root, data_root, destination):
    if json.loads((root/"status.json").read_text())["stage"] != "complete":
        raise RuntimeError("Training and all final tests must finish first")
    write_json(root/"finalization.json", {"stage": "diagnostics"})
    diagnose(root, data_root)
    write_json(root/"finalization.json", {"stage": "cost_measurement"})
    measure(root/"regularizer-cost.json")
    summarize(root)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("REPORT.md", "summary.json", "manifest.json", "diagnostics.json", "regularizer-cost.json"):
        shutil.copy2(root/filename, destination/filename)
    for env in json.loads((root/"manifest.json").read_text())["settings"]["environments"]:
        target = destination/env
        target.mkdir(exist_ok=True)
        for filename in ("partition.json", "data-check.json", "validation.json", "test.json", "selected.json", "random-test.json"):
            shutil.copy2(root/env/filename, target/filename)
    write_json(root/"finalization.json", {"stage": "complete", "report": str(destination/"REPORT.md")})
    print((destination/"REPORT.md").read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/two-environment-benchmark"))
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument("--destination", type=Path, default=Path("experiments/results/two-environment-screen"))
    args = parser.parse_args()
    try:
        finalize(args.root, args.data_root, args.destination)
    except Exception as error:
        write_json(args.root/"finalization.json", {"stage": "error", "error": repr(error)})
        raise
