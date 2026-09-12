"""Post-run spread diagnostics on a fixed sample of reserved test clips.

These diagnostics never feed coefficient selection. Model seeds share exactly
the same 512 clips; correlated frames are not treated as independent trials.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from run import Clips, ENVIRONMENTS, write_json
from runtime import build_model, normalize_pixels


@torch.no_grad()
def diagnose(root, data_root):
    if json.loads((root/"status.json").read_text())["stage"] != "complete":
        raise RuntimeError("Diagnostics are post-run only")
    torch.set_num_threads(4)
    rows = []
    for env in json.loads((root/"manifest.json").read_text())["settings"]["environments"]:
        folder = root/env
        partition = json.loads((folder/"partition.json").read_text())
        tests = json.loads((folder/"test.json").read_text())
        path = data_root/"datasets"/ENVIRONMENTS[env][0]
        path = path.with_name(path.stem+"_frames.h5")
        clips = Clips(path, partition["test"], 8802, 8, 64)
        for run in tests:
            name = f"{run['kind']}-w{run['weight']}-s{run['seed']}"
            output = folder/name/"diagnostics.json"
            if output.exists():
                rows.append(json.loads(output.read_text()))
                continue
            model = build_model(len(partition["action_mean"])).cuda().eval()
            model.load_state_dict(torch.load(folder/name/"weights.pt", map_location="cuda", weights_only=True))
            embeddings = []
            for batch in DataLoader(clips, batch_size=64, num_workers=0):
                z = model.encode({"pixels": normalize_pixels(batch["pixels"].cuda())})["emb"]
                # One first-frame embedding per clip avoids multiplying the
                # sample size by strongly dependent adjacent frames.
                embeddings.append(z[:, 0].cpu().double())
            z = torch.cat(embeddings)
            centered = z-z.mean(0)
            covariance = centered.T@centered/(len(z)-1)
            std = covariance.diagonal().sqrt()
            corr = covariance/std.clamp_min(1e-12)[:, None]/std.clamp_min(1e-12)[None, :]
            eigen = torch.linalg.eigvalsh(covariance).clamp_min(0)
            p = eigen/eigen.sum().clamp_min(1e-30)
            entropy_rank = float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum()))
            mask = ~torch.eye(z.shape[-1], dtype=bool)
            row = {"environment": env, "kind": run["kind"], "weight": run["weight"], "seed": run["seed"],
                   "clips": len(z), "std_mean": float(std.mean()), "std_min": float(std.min()),
                   "covariance_entropy_rank": entropy_rank,
                   "mean_absolute_correlation": float(corr[mask].abs().mean())}
            write_json(output, row)
            rows.append(row)
            del model
            torch.cuda.empty_cache()
    write_json(root/"diagnostics.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/two-environment-benchmark"))
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    args = parser.parse_args()
    diagnose(args.root, args.data_root)
