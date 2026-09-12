"""Direct regularizer study: fixed architecture/width/loss, no decoder.

Small synthetic pilot, NOT the published LeWM/TwoRoom benchmark. Validation
planning chooses one coefficient per family; a separate test set is evaluated
only after that choice. Simulator position is never a training target.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch import nn

from pilot import TinyEncoder, dataset
from lewm.jepa import JEPA
from lewm.module import ARPredictor, Embedder, MLP
from lewm.regularizers import DistributionRegularizer, make_regularizer


WIDTH = 192
GRID = {"gaussian": [.03, .09, .27], "mixture": [.03, .09, .27],
        "matched_gaussian": [.03, .09, .27], "moments": [.3, 3., 30.], "none": [0.]}


def model():
    # Match upstream projector normalization, but use a small CNN and smaller
    # hidden predictor for this pilot. These settings are identical in EVERY arm.
    return JEPA(encoder=TinyEncoder(), projector=MLP(64, 128, WIDTH, norm_fn=nn.BatchNorm1d),
                predictor=ARPredictor(input_dim=WIDTH, hidden_dim=64, output_dim=WIDTH,
                                      num_frames=3, depth=2, heads=4, dim_head=16, mlp_dim=128),
                action_encoder=Embedder(2, emb_dim=WIDTH),
                pred_proj=MLP(WIDTH, 128, WIDTH, norm_fn=nn.BatchNorm1d))


def training_loss(m, batch, reg, weight):
    out = m.encode({k: v[:, :4] for k, v in batch.items()})
    z, actions = out["emb"], out["act_emb"]
    prediction = m.predict(z[:, :3], actions[:, :3])
    mse = (prediction - z[:, 1:4]).square().mean()
    penalty = reg(z.transpose(0, 1)) if reg is not None and weight else mse.new_zeros(())
    return mse + weight * penalty


def transition(pos, action):
    nxt = (pos + action).clamp(-.9, .9)
    crossing = (pos[..., 0] * nxt[..., 0] <= 0) & (nxt[..., 1].abs() > .18)
    nxt[..., 0] = torch.where(crossing, pos[..., 0], nxt[..., 0])
    return nxt


@torch.no_grad()
def planning(m, data, states, *, episodes=24, candidates=64, iterations=2, seed=551):
    """Equal-budget open-loop CEM, horizon five, Euclidean latent goal cost.

    Plans are scored by the model. Physics is used only AFTER selecting a plan.
    Exclude already-near goals using a fixed model-independent threshold.
    """
    chosen = ((states[:, 7]-states[:, 2]).norm(dim=-1) > .2).nonzero().flatten()[:episodes]
    if len(chosen) < episodes:
        raise ValueError("Insufficient nontrivial planning episodes")
    b = {k: v[chosen] for k, v in data.items()}
    z = m.encode(dict(b))["emb"]
    current = z[:, :3, None].transpose(1, 2).expand(-1, candidates, -1, -1).reshape(-1, 3, WIDTH)
    goal = z[:, 7, None]
    past = b["action"][:, :2, None].transpose(1, 2).expand(-1, candidates, -1, -1).reshape(-1, 2, 2)
    mean = torch.zeros(episodes, 5, 2, device=z.device)
    std = torch.full_like(mean, .18)
    gen = torch.Generator(device=z.device).manual_seed(seed)
    best_plan = torch.zeros_like(mean)
    for iteration in range(iterations):
        if iteration == 0:
            plans = torch.rand(episodes, candidates, 5, 2, device=z.device, generator=gen) * .36 - .18
        else:
            plans = (mean[:, None] + std[:, None] * torch.randn(
                episodes, candidates, 5, 2, device=z.device, generator=gen)).clamp(-.18, .18)
            plans[:, 0] = best_plan  # retain previous best candidate
        actions = torch.cat((past, plans.reshape(-1, 5, 2)), dim=1)
        action_emb = m.action_encoder(actions)
        ctx = current
        for k in range(5):
            pred = m.predict(ctx[:, -3:], action_emb[:, k:k+3])[:, -1:]
            ctx = torch.cat((ctx, pred), 1)
        cost = (ctx[:, -1].reshape(episodes, candidates, WIDTH) - goal).square().sum(-1)
        elite = cost.topk(8, largest=False).indices
        elite_plans = plans.gather(1, elite[..., None, None].expand(-1, -1, 5, 2))
        best_plan = elite_plans[:, 0]
        mean, std = elite_plans.mean(1), elite_plans.std(1).clamp_min(.02)
    pos = states[chosen, 2].clone()
    random_pos = pos.clone()
    random_gen = torch.Generator(device=z.device).manual_seed(761)
    random_plan = torch.rand(episodes, 5, 2, device=z.device, generator=random_gen) * .36 - .18
    for k in range(5):
        pos = transition(pos, best_plan[:, k])
        random_pos = transition(random_pos, random_plan[:, k])
    distance = (pos - states[chosen, 7]).norm(dim=-1)
    return {"goal_distance": float(distance.mean()), "success_rate": float((distance < .12).float().mean()),
            "stay_distance": float((states[chosen, 2]-states[chosen, 7]).norm(dim=-1).mean()),
            "random_distance": float((random_pos-states[chosen, 7]).norm(dim=-1).mean()),
            "planning_episodes": episodes, "per_episode_distance": distance.cpu().tolist()}


@torch.no_grad()
def diagnostics(m, train, train_states, data, states):
    m.eval()
    z_train = m.encode(dict(train))["emb"].flatten(0, 1).double()
    out = m.encode(dict(data))
    z = out["emb"]
    mean, std = z_train.mean(0), z_train.std(0).clamp_min(1e-6)
    def design(x):
        x = (x.double() - mean) / std
        return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
    x_train = design(z_train)
    ridge = torch.eye(WIDTH+1, device=z.device, dtype=torch.float64) * .1
    ridge[-1, -1] = 0
    coef = torch.linalg.solve(x_train.T @ x_train + ridge,
                             x_train.T @ train_states.flatten(0, 1).double())
    probe = (design(z) @ coef - states).square().mean()
    ctx = z[:, :3]
    shuffled = z[:, :3]
    shuffle_actions = m.action_encoder(data["action"].roll(1, 0))
    for k in range(5):
        pred = m.predict(ctx[:, -3:], out["act_emb"][:, k:k+3])[:, -1:]
        shuffled_pred = m.predict(shuffled[:, -3:], shuffle_actions[:, k:k+3])[:, -1:]
        ctx, shuffled = torch.cat((ctx, pred), 1), torch.cat((shuffled, shuffled_pred), 1)
    rollout_probe = (design(ctx[:, -1]) @ coef - states[:, 7]).square().mean()
    shuffled_probe = (design(shuffled[:, -1]) @ coef - states[:, 7]).square().mean()
    eigen = torch.linalg.eigvalsh(torch.cov(z.flatten(0, 1).double().T)).clamp_min(0)
    p = eigen / eigen.sum().clamp_min(1e-30)
    rank = (-(p*p.clamp_min(1e-30).log()).sum()).exp()
    axis = z.mean(-1).flatten()
    hist = torch.histc(axis, bins=32, min=-2, max=2)
    return {"position_probe_mse": float(probe), "future_position_probe_mse": float(rollout_probe),
            "action_shuffle_probe_delta": float(shuffled_probe - rollout_probe),
            "latent_variance": float(eigen.sum()), "effective_rank": float(rank),
            "mixture_positive_fraction": float((axis>0).float().mean()),
            "mixture_valley_fraction": float((axis.abs()<.3).float().mean()),
            "mixture_center_fraction": float(((axis.abs()-math.sqrt(.75)).abs()<.2).float().mean()),
            "mixture_axis_histogram": hist.cpu().tolist()}


def run(args, *, grid=None, test_seed=36000, extra_diagnostics=None, sources=(), fixed_comparison=False,
        select_primary=False):
    grid = GRID if grid is None else grid
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    def get(seed, count):
        b, states = dataset(seed, count)
        return {k: v.to(device) for k, v in b.items()}, states.to(device)
    train, train_states = get(12000, 256)
    val, val_states = get(24000, 128)
    report = {"scope": "fixed-width synthetic pilot; no decoder; not published TwoRoom",
              "width": WIDTH, "steps": args.steps, "grid": grid, "seeds": args.seeds,
              "data_seeds": {"train": 12000, "validation": 24000, "test": test_seed},
              "selection": ("fixed targets and coefficients; no selection" if fixed_comparison else
                            "minimum mean validation CEM goal distance across initialization seeds"),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "torch": torch.__version__, "device": device,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                  Path(__file__), Path("src/lewm/regularizers.py"), Path("experiments/pilot.py"), *sources]},
              "validation": [], "test": []}
    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
    for seed in args.seeds:
        for kind, weights in grid.items():
            for weight in weights:
                torch.manual_seed(seed)
                m = model().to(device)
                reg = make_regularizer(kind, latent_dim=WIDTH)
                if reg is not None:
                    reg = reg.to(device)
                opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
                gen = torch.Generator().manual_seed(seed+500)
                start = time.perf_counter()
                for step in range(args.steps):
                    m.train()
                    idx = torch.randperm(256, generator=gen)[:64].to(device)
                    b = {k: v[idx] for k, v in train.items()}
                    opt.zero_grad(set_to_none=True)
                    loss = training_loss(m, b, reg, weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss {kind} {weight} {seed}")
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.)
                    opt.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                m.eval()
                metrics = {**diagnostics(m, train, train_states, val, val_states), **planning(m, val, val_states)}
                if extra_diagnostics is not None:
                    metrics.update(extra_diagnostics(m, val))
                row = dict(kind=kind, weight=weight, seed=seed, train_seconds=seconds, **metrics)
                report["validation"].append(row)
                torch.save(m.state_dict(), args.output / f"{kind}-w{weight}-s{seed}.pt")
                save()
                print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}), flush=True)
    selected = {}
    for kind, weights in grid.items():
        selected[kind] = min(weights, key=lambda w: statistics.mean(
            r["goal_distance"] for r in report["validation"] if r["kind"] == kind and r["weight"] == w))
    report["selected_weights"] = selected
    if select_primary:
        report["selected_primary_kind"] = min(grid, key=lambda kind: statistics.mean(
            r["goal_distance"] for r in report["validation"]
            if r["kind"] == kind and r["weight"] == selected[kind]))
    student_kinds = [kind for kind in grid if kind.startswith("student_t")]
    if student_kinds and not fixed_comparison:
        report["selected_student_kind"] = min(student_kinds, key=lambda kind: statistics.mean(
            r["goal_distance"] for r in report["validation"]
            if r["kind"] == kind and r["weight"] == selected[kind]))
    save()
    # Only now load the independent test trajectories. No coefficient changes after this.
    test, test_states = get(test_seed, 128)
    for kind, weight in selected.items():
        for seed in args.seeds:
            m = model().to(device).eval()
            m.load_state_dict(torch.load(args.output / f"{kind}-w{weight}-s{seed}.pt", map_location=device, weights_only=True))
            row = dict(kind=kind, weight=weight, seed=seed,
                       **diagnostics(m, train, train_states, test, test_states), **planning(m, test, test_states))
            if extra_diagnostics is not None:
                row.update(extra_diagnostics(m, test))
            report["test"].append(row)
            save()
    print("DONE: " + str(args.output / "results.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("outputs/regularizer-study"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    run(args)
