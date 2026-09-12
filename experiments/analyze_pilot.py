"""Evaluate fixed checkpoints with constant-image, action and ablation controls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

from pilot import dataset, make_model
from lewm.bottleneck import predictive_loss, multiscale_error


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads((args.directory / "results.json").read_text())
    train, _ = dataset(12000, 256)
    val, _ = dataset(24000, 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = ({k: v.to(device) for k, v in b.items()} for b in (train, val))
    constant = sum(multiscale_error(train["pixels"][:, 2+k].mean(0, keepdim=True),
                                   val["pixels"][:, 2+k]).mean() for k in (1, 3, 5)) / 3
    for row in report["results"]:
        model = make_model(row["width"]).to(device).eval()
        path = args.directory / f'{row["variant"]}-d{row["width"]}-s{row["seed"]}.pt'
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        base = predictive_loss(model, val, None)["obs_loss"]
        shuffled = dict(val, action=val["action"].roll(1, dims=0))
        row["action_shuffle_delta"] = float(predictive_loss(model, shuffled, None)["obs_loss"] - base)
        variance = model.encode(dict(train))["emb"].flatten(0, 1).var(0)
        weak = variance.argsort()[:max(1, row["width"] // 4)]
        def ablate(module, inputs, output):
            z = output.clone()
            z[:, weak] = 0
            return z
        hook = model.projector.register_forward_hook(ablate)
        try:
            row["weak_quarter_ablation_delta"] = float(predictive_loss(model, val, None)["obs_loss"] - base)
        finally:
            hook.remove()
        row["constant_image_obs_loss"] = float(constant)
    report["experiment_source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path("experiments/pilot.py"), Path("src/lewm/bottleneck.py"),
                     Path("src/lewm/module.py"), Path("src/lewm/jepa.py")]
    }
    (args.directory / "analyzed.json").write_text(json.dumps(report, indent=2))
    lines = ["# Synthetic predictive-bottleneck pilot", "",
             "Not the published TwoRoom benchmark. Small CNN, upstream transformer/SIGReg; "
             f"256 training and 64 held-out synthetic episodes; updates per run: {sorted({r['steps'] for r in report['results']})}.", "",
             f"Initialization seeds: {sorted({r['seed'] for r in report['results']})}. "
             "Values are mean ± sample standard deviation across initialization seeds. "
             "Lower errors are better. State is used only for diagnostic probes.", "",
             "| Width | Variant | Future image MSE | Position probe MSE | Latent variance |",
             "|---|---|---|---|---|"]
    for width in sorted({r["width"] for r in report["results"]}):
        for variant in dict.fromkeys(r["variant"] for r in report["results"]):
            rows = [r for r in report["results"] if r["width"] == width and r["variant"] == variant]
            def fmt(key):
                values = [r[key] for r in rows]
                return f"{statistics.mean(values):.4g} ± {statistics.stdev(values):.2g}" if len(values) > 1 else f"{values[0]:.4g}"
            lines.append(f"| {width} | {variant} | {fmt('obs_loss')} | {fmt('position_probe_mse')} | {fmt('latent_variance')} |")
    lines += ["", f"Training-mean image baseline MSE: {float(constant):.6g}. "
              f"Last-frame persistence MSE: {report['results'][0]['persistence_obs_loss']:.6g}.", "",
              "The no-SIGReg variant shows very small latent variance and poor position probes. "
              "Low whole-image error alone does not establish useful dynamics. "
              "Effective rank must be interpreted alongside absolute variance.", "",
              "The baseline decoder is trained on detached forecasts, with separately clipped gradients. "
              "compact_low_beta controls for the latent coefficient change from 1 to 0.1. "
              "All models use the same data and minibatch schedules. The synthetic encoder/projector "
              "uses LayerNorm and reduced hidden capacity; the full experiment retains the upstream "
              "ViT and BatchNorm projectors. These are pipeline findings, not a claim about LeWM benchmark performance.", "",
              "analyzed.json includes each seed, training cost, rank, constant-image control, "
              "action-shuffle sensitivity, and zero-ablation of the least-variable quarter of input features. "
              "Ablation is coordinate-dependent and is not evidence of a uniquely meaningful subspace. "
              "No planning-success result has been measured."]
    lines += ["", "Mean increase in observation MSE after shuffling actions between held-out episodes:", ""]
    for variant in dict.fromkeys(r["variant"] for r in report["results"]):
        delta = statistics.mean(r["action_shuffle_delta"] for r in report["results"] if r["variant"] == variant)
        lines.append(f"- {variant}: {delta:.6g}")
    lines += ["", "![Fixed held-out forecast examples](predictions.png)"]
    (args.directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
