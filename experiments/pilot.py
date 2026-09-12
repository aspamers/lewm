"""Small synthetic dynamics pilot. NOT the published TwoRoom benchmark.

Run from the repository root with PYTHONPATH=src. Uses upstream JEPA,
ARPredictor, MLP, Embedder and SIGReg, with a small CNN replacing the ViT.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from lewm.bottleneck import (BottleneckPredictor, ObservationDecoder,
                             PredictiveBottleneck, predictive_loss, multiscale_error)
from lewm.module import MLP, Embedder, SIGReg


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(),
                                 nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(), nn.Flatten(),
                                 nn.Linear(32 * 8 * 8, 64))

    def forward(self, x, **kwargs):
        return SimpleNamespace(last_hidden_state=self.net(x)[:, None])


def dataset(seed, episodes):
    """Independent short episodes of an action-driven spot, with a wall/door."""
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(episodes, 2, generator=gen) * 1.6 - .8
    action = (torch.rand(episodes, 8, 2, generator=gen) * 2 - 1) * .18
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
    wall = ((xx.abs() < .05) & (yy.abs() > .22)).float()
    images, states = [], []
    for t in range(8):
        states.append(pos.clone())
        spot = torch.exp(-((xx-pos[:, 0, None, None])**2 + (yy-pos[:, 1, None, None])**2) / .012)
        images.append(torch.stack((spot, wall.expand_as(spot), .2 * spot), dim=1))
        nxt = (pos + action[:, t]).clamp(-.9, .9)
        crossing = (pos[:, 0] * nxt[:, 0] <= 0) & (nxt[:, 1].abs() > .18)
        nxt[crossing, 0] = pos[crossing, 0]
        pos = nxt
    return {"pixels": torch.stack(images, 1), "action": action}, torch.stack(states, 1)


def make_model(width):
    return PredictiveBottleneck(
        encoder=TinyEncoder(), projector=MLP(64, 128, width),
        predictor=BottleneckPredictor(width, hidden_dim=64, num_frames=3,
                                      depth=2, heads=4, dim_head=16, mlp_dim=128),
        action_encoder=Embedder(2, emb_dim=64), pred_proj=MLP(64, 128, width),
        decoder=ObservationDecoder(width, 32, hidden_dim=16))


@torch.no_grad()
def evaluate(m, data, states, probe_train, train_states):
    m.eval()
    losses = predictive_loss(m, data, None)
    z = m.encode(dict(data))["emb"].flatten(0, 1).double()
    z_train = m.encode(dict(probe_train))["emb"].flatten(0, 1).double()
    mean, std = z_train.mean(0), z_train.std(0).clamp_min(1e-6)
    def design(x):
        x = (x - mean) / std
        return torch.cat((x, torch.ones_like(x[:, :1])), 1)
    x, xt = design(z), design(z_train)
    ridge = torch.eye(xt.shape[1], device=z.device, dtype=z.dtype) * .01
    ridge[-1, -1] = 0
    coef = torch.linalg.solve(xt.T @ xt + ridge, xt.T @ train_states.flatten(0, 1).double())
    mse = (x @ coef - states.flatten(0, 1)).square().mean()
    eigen = torch.linalg.eigvalsh(torch.cov(z.T)).clamp_min(0)
    p = eigen / eigen.sum().clamp_min(1e-30)
    rank = (-(p * p.clamp_min(1e-30).log()).sum()).exp()
    persistence = sum(multiscale_error(data["pixels"][:, 2], data["pixels"][:, 2+k]).mean()
                      for k in (1, 3, 5)) / 3
    return {**{k: float(v) for k, v in losses.items() if k.startswith("obs")},
            "position_probe_mse": float(mse), "effective_rank": float(rank),
            "latent_variance": float(eigen.sum()), "persistence_obs_loss": float(persistence)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot"))
    args = parser.parse_args()
    if args.steps < 1 or min(args.widths) < 1:
        parser.error("steps and widths must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, train_states = dataset(12000, 256)
    val, states = dataset(24000, 64)
    train, val = ({k: v.to(device) for k, v in d.items()} for d in (train, val))
    train_states, states = train_states.to(device), states.to(device)
    results = []
    for seed in args.seeds:
        for width in args.widths:
            for variant, obs_weight, sig_weight, beta in [
                ("compact_lewm", 0., .09, 1.),
                ("compact_low_beta", 0., .09, .1),
                ("predictive", 1., 0., .1),
                ("predictive_sigreg", 1., .09, .1),
            ]:
                torch.manual_seed(seed)
                m = make_model(width).to(device)
                reg = SIGReg(num_proj=64).to(device)
                optimizer = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
                # Same minibatch schedule independent of SIGReg's random projections.
                gen = torch.Generator().manual_seed(seed + 500)
                start = time.perf_counter()
                for step in range(args.steps):
                    m.train()
                    idx = torch.randperm(256, generator=gen)[:32].to(device)
                    b = {k: v[idx] for k, v in train.items()}
                    optimizer.zero_grad(set_to_none=True)
                    loss = predictive_loss(m, b, reg, observation_weight=obs_weight,
                                           sigreg_weight=sig_weight, latent_weight=beta)["loss"]
                    # A separate diagnostic decoder is fit on detached forecasts
                    # for the reconstruction-free baseline; no encoder gradient.
                    if not obs_weight:
                        with torch.no_grad():
                            out = m.encode(dict(b))
                            ctx = out["emb"][:, :3]
                            forecasts = []
                            for k in range(1, 6):
                                pred = m.predict(ctx[:, -3:], out["act_emb"][:, k-1:k+2])[:, -1:]
                                ctx = torch.cat((ctx, pred), 1)
                                if k in (1, 3, 5):
                                    forecasts.append((k, pred[:, 0].detach()))
                        diagnostic = sum(multiscale_error(m.decoder(p), b["pixels"][:, 2+k]).mean()
                                         for k, p in forecasts) / 3
                        loss = loss + diagnostic
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    loss.backward()
                    # Clip decoder separately so diagnostic gradients cannot
                    # change the baseline's encoder/predictor update magnitude.
                    nn.utils.clip_grad_norm_([p for n, p in m.named_parameters() if not n.startswith("decoder.")], 1.)
                    nn.utils.clip_grad_norm_(m.decoder.parameters(), 1.)
                    optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                metrics = evaluate(m, val, states, train, train_states)
                row = dict(seed=seed, width=width, variant=variant, steps=args.steps,
                           train_seconds=seconds, parameters=sum(p.numel() for p in m.parameters()), **metrics)
                results.append(row)
                print(json.dumps(row), flush=True)
                torch.save(m.state_dict(), args.output / f"{variant}-d{width}-s{seed}.pt")
                (args.output / "results.json").write_text(json.dumps({
                    "scope": "synthetic pilot, CNN encoder, not benchmark evidence",
                    "torch": torch.__version__, "python": platform.python_version(),
                    "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "data_seeds": [12000, 24000], "results": results}, indent=2))


if __name__ == "__main__":
    main()
