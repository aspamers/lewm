"""Bounded real TwoRoom/PushT comparison using the repo's full architecture.

Train every candidate before final-test evaluation. Resume completed candidates,
and retain optimizer/RNG state for interrupted training. Scientific settings are
recorded in the manifest and cannot be changed in an existing output directory.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader, Dataset

from partitions import make_partition, EvaluationSeeds
from check_data import check as check_data
from rechunk import convert
from runtime import build_model, normalize_pixels, objective
from calibration import calibrate_projection_bn
from lewm.regularizers import make_regularizer, VarianceFloorRegularizer

ENVIRONMENTS = {"tworoom": ("tworoom.h5", "swm/TwoRoom-v1"),
                "pusht": ("pusht_expert_train.h5", "swm/PushT-v1")}
GRID = {"gaussian": [.03, .09, .27], "variance_decorrelation": [.03, .1, .3]}
FIXED = {"gaussian": .09, "variance_decorrelation": .3}


class PairedWorld(swm.World):
    def reset(self, seed=None, options=None):
        # SWM's dataset extraction returns NumPy seeds, while Gymnasium requires
        # native Python integers for each environment reset.
        if isinstance(seed, np.ndarray):
            seed = [int(value) for value in seed.reshape(-1)]
        elif isinstance(seed, np.generic):
            seed = int(seed)
        return super().reset(seed=seed, options=options)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


class Clips(Dataset):
    def __init__(self, path, episodes, seed, steps, batch, start_step=0):
        self.path = str(path)
        self.handle = None
        with h5py.File(path, "r") as f:
            self.lengths, self.offsets = f["ep_len"][:], f["ep_offset"][:]
        self.episodes = np.array(episodes)
        # Match the distribution of uniform sampling over eligible clips.
        self.counts = self.lengths[self.episodes]-19
        rng = np.random.default_rng(seed)
        chosen = rng.choice(self.episodes, size=steps*batch, p=self.counts/self.counts.sum())
        starts = np.array([rng.integers(self.lengths[ep]-19) for ep in chosen])
        self.chosen, self.starts = chosen[start_step*batch:], starts[start_step*batch:]

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r", rdcc_nbytes=64*1024**2)
        start = int(self.offsets[self.chosen[index]]+self.starts[index])
        # HDF5's strided hyperslab selection is much slower than four whole
        # frame reads, even with one-frame chunks. These observations are exact.
        pixels = np.stack([self.handle["pixels"][start+i] for i in (0, 5, 10, 15)])
        if pixels.shape[-1] == 3:
            pixels = pixels.transpose(0, 3, 1, 2)
        return {"pixels": torch.from_numpy(pixels.copy()),
                "action": torch.from_numpy(self.handle["action"][start:start+20].copy())}


def prepare_data(path, output):
    manifest = output / "partition.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    with h5py.File(path, "r") as f:
        lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
        partition = make_partition(lengths)
        count = 0
        total = squares = None
        for ep in partition["train"]:
            x = f["action"][int(offsets[ep]):int(offsets[ep]+lengths[ep])].astype(np.float64)
            x = x[np.isfinite(x).all(-1)]
            total = x.sum(0) if total is None else total+x.sum(0)
            squares = (x*x).sum(0) if squares is None else squares+(x*x).sum(0)
            count += len(x)
        partition.update(action_mean=(total/count).tolist(),
                         action_std=np.sqrt(np.maximum((squares-total*total/count)/(count-1), 1e-12)).tolist(),
                         columns=list(f.keys()), total_episodes=len(lengths))
    write_json(manifest, partition)
    return partition


def train(path, partition, kind, weight, seed, args, output):
    complete, resume = output/"training.json", output/"resume.pt"
    if complete.exists():
        return json.loads(complete.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    precision = getattr(args, "precision", "float32")
    if precision not in ("float32", "bfloat16"):
        raise ValueError(f"Unsupported precision: {precision}")
    # Avoid reduced-precision representation spread being mistaken for signal.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(len(partition["action_mean"])).cuda().train()
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    regularizer_parameters = {}
    if kind == "variance_decorrelation" and hasattr(args, "floor_weight"):
        regularizer_parameters = {"floor_weight":args.floor_weight,"correlation_weight":args.correlation_weight}
        reg = VarianceFloorRegularizer(True, **regularizer_parameters).cuda()
    else:
        reg = make_regularizer(kind, num_proj=1024).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    step0, elapsed = 0, 0.
    if resume.exists():
        state = torch.load(resume, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        step0, elapsed = state["step"], state["elapsed"]
        del state
    clips = Clips(path, partition["train"], seed+500, args.steps, args.batch, step0)
    loader = DataLoader(clips, batch_size=args.batch, num_workers=2, pin_memory=True,
                        generator=torch.Generator().manual_seed(seed+700))
    mean = torch.tensor(partition["action_mean"], device="cuda")
    std = torch.tensor(partition["action_std"], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step, batch in enumerate(loader, start=step0+1):
        pixels = normalize_pixels(batch["pixels"].cuda(non_blocking=True))
        actions = batch["action"].cuda(non_blocking=True)
        # Match the four-frame one-step objective: terminal NaNs are replaced
        # after raw normalization; boundary padding is not a prediction target.
        actions = torch.nan_to_num((actions-mean)/std).reshape(args.batch, 4, -1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            loss, prediction, penalty = objective(model, {"pixels": pixels, "action": actions},
                                                  reg, weight, seed*100000+step)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at {output}, step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            torch.cuda.synchronize()
            progress = {"step": step, "loss": float(loss), "prediction_loss": float(prediction),
                        "regularizer_loss": float(penalty), "seconds": elapsed+time.perf_counter()-start}
            write_json(output/"progress.json", progress)
            print(str(output), progress, flush=True)
        if step % 100 == 0 or step == args.steps:
            temporary = output/"resume.tmp"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "elapsed": elapsed+time.perf_counter()-start, "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state()}, temporary)
            temporary.replace(resume)
    torch.cuda.synchronize()
    torch.save(model.state_dict(), output/"weights.pt")
    result = {"kind": kind, "weight": weight, "seed": seed, "steps": args.steps,
              "precision": precision, "encoder_gradient_checkpointing": True,
              "regularizer_parameters":regularizer_parameters,
              "train_seconds": elapsed+time.perf_counter()-start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "parameters": sum(p.numel() for p in model.parameters())}
    write_json(complete, result)
    del model, reg, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def evaluate(path, partition, env, checkpoint, stage, args, calibrate=True):
    destination = checkpoint/(stage+("-calibrated" if calibrate else "")+".json")
    provenance = None
    if calibrate:
        provenance = {"split":"train","seed":8821,"clips":512,"batch":64,
                      "evaluation":{"stage":stage,"goals":partition[stage+"_goals"],
                                    "candidates":args.candidates,"cem_steps":args.cem_steps},
                      "evaluation_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "checkpoint_sha256":hashlib.sha256((checkpoint/"weights.pt").read_bytes()).hexdigest(),
                      "train_episodes_sha256":hashlib.sha256(json.dumps(partition["train"]).encode()).hexdigest(),
                      "code_sha256":hashlib.sha256(Path(__file__).with_name("calibration.py").read_bytes()).hexdigest()}
    if destination.exists():
        previous = json.loads(destination.read_text())
        if calibrate and previous.get("calibration",{}).get("provenance") != provenance:
            raise ValueError("Calibration inputs changed; use a fresh evaluation output")
        return previous
    torch.manual_seed(6173)
    model = build_model(len(partition["action_mean"])).cuda().eval()
    model.load_state_dict(torch.load(checkpoint/"weights.pt", map_location="cuda", weights_only=True))
    model.requires_grad_(False)
    calibration = None
    if calibrate:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cache = path.with_name(path.stem+"_frames.h5")
        loader = DataLoader(Clips(cache, partition["train"],8821,8,64),batch_size=64,
                            generator=torch.Generator().manual_seed(8821))
        calibration = {"provenance":provenance, **calibrate_projection_bn(model,
                       ({"pixels":normalize_pixels(b["pixels"].cuda())} for b in loader))}
        del loader
    scaler = StandardScaler()
    scaler.mean_ = np.array(partition["action_mean"])
    scaler.scale_ = np.array(partition["action_std"])
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)
    dataset = EvaluationSeeds(swm.data.HDF5Dataset(path=path))
    goal = partition[stage+"_goals"]
    world = PairedWorld(ENVIRONMENTS[env][1], num_envs=len(goal["episodes"]),
                      image_shape=(224, 224), max_episode_steps=100)
    solver = swm.solver.CEMSolver(model=model, batch_size=1, num_samples=args.candidates,
                                 n_steps=args.cem_steps, topk=8, device="cuda", seed=6173)
    world.set_policy(swm.policy.WorldModelPolicy(solver=solver,
                     config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
                     process={"action": scaler}, transform={"pixels": normalize_pixels, "goal": normalize_pixels}))
    config = OmegaConf.load(Path(__file__).resolve().parents[2]/f"config/eval/{env}.yaml")
    start = time.perf_counter()
    try:
        metrics = world.evaluate(dataset=dataset, episodes_idx=goal["episodes"], start_steps=goal["starts"],
                                 goal_offset=25, eval_budget=50,
                                 callables=OmegaConf.to_container(config.eval.callables, resolve=True))
    finally:
        world.close()
    results = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}
    results.update(evaluation_seconds=time.perf_counter()-start, goals=goal,
                   generated_evaluation_seeds=dataset.injected)
    if calibration is not None:
        results["calibration"] = calibration
    write_json(destination, results)
    del model, solver, world
    gc.collect()
    torch.cuda.empty_cache()
    return results


def random_control(path, partition, env, output, args):
    destination = output/"random-test.json"
    if destination.exists():
        return
    dataset = EvaluationSeeds(swm.data.HDF5Dataset(path=path))
    goal = partition["test_goals"]
    world = PairedWorld(ENVIRONMENTS[env][1], num_envs=len(goal["episodes"]),
                      image_shape=(224, 224), max_episode_steps=100)
    world.set_policy(swm.policy.RandomPolicy(seed=6173))
    config = OmegaConf.load(Path(__file__).resolve().parents[2]/f"config/eval/{env}.yaml")
    try:
        metrics = world.evaluate(dataset=dataset, episodes_idx=goal["episodes"], start_steps=goal["starts"],
                                 goal_offset=25, eval_budget=50,
                                 callables=OmegaConf.to_container(config.eval.callables, resolve=True))
    finally:
        world.close()
    write_json(destination, {k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/two-environment-benchmark"))
    parser.add_argument("--environments", nargs="+", choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 32])
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--cem-steps", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {**vars(args), "root": str(args.root), "output": str(args.output), "grid": GRID, "fixed": FIXED}
    manifest = args.output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["settings"] != settings:
        raise ValueError("Cannot change settings in an existing run directory")
    sources = [Path(__file__), Path(__file__).with_name("runtime.py"), Path(__file__).with_name("partitions.py"),
               Path(__file__).with_name("check_data.py"), Path(__file__).with_name("rechunk.py"),
               Path(__file__).with_name("calibration.py"), Path("src/lewm/regularizers.py"),
               Path("src/lewm/jepa.py"), Path("src/lewm/module.py"),
               Path("config/train/model/lewm.yaml"), Path("config/eval/tworoom.yaml"), Path("config/eval/pusht.yaml")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if manifest.exists() and json.loads(manifest.read_text())["sources"] != hashes:
        raise ValueError("Source code changed; use a fresh output directory to avoid mixing protocols")
    if not manifest.exists():
        write_json(manifest, {"settings": settings, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                             "sources": hashes})
    for env in args.environments:
        path = args.root/"datasets"/ENVIRONMENTS[env][0]
        out = args.output/env
        out.mkdir(exist_ok=True)
        while not path.with_suffix(".verified.json").exists():
            download_status = args.root/(env+"-download.json")
            if download_status.exists() and json.loads(download_status.read_text()).get("stage") == "error":
                raise RuntimeError(f"Dataset preparation failed: {download_status.read_text()}")
            write_json(args.output/"status.json", {"stage": "waiting_for_verified_dataset", "environment": env})
            time.sleep(10)
        data_check = check_data(path, env, out)
        if data_check["pixel_mse"] > .02:
            raise RuntimeError("Recorded and simulator images differ materially; inspect data-check images")
        partition = prepare_data(path, out)
        training_path = path.with_name(path.stem+"_frames.h5")
        if not training_path.with_suffix(".cache.json").exists():
            write_json(args.output/"status.json", {"stage": "lossless_training_cache", "environment": env})
            convert(path, training_path)
        validation = []
        for seed in args.seeds:
            for kind, weights in GRID.items():
                for weight in weights:
                    checkpoint = out/f"{kind}-w{weight}-s{seed}"
                    write_json(args.output/"status.json", {"stage": "training_and_validation", "run": str(checkpoint)})
                    training = train(training_path, partition, kind, weight, seed, args, checkpoint)
                    metrics = evaluate(path, partition, env, checkpoint, "validation", args)
                    validation.append({**training, **metrics})
                    write_json(out/"validation.json", validation)
        selected = {kind: max(weights, key=lambda w: np.mean([r["success_rate"] for r in validation
                            if r["kind"] == kind and r["weight"] == w])) for kind, weights in GRID.items()}
        # Freeze per-environment selection before opening final-test outcomes.
        write_json(out/"selected.json", selected)
        random_control(path, partition, env, out, args)
        results = []
        for kind in GRID:
            for weight in sorted({selected[kind], FIXED[kind]}):
                for seed in args.seeds:
                    checkpoint = out/f"{kind}-w{weight}-s{seed}"
                    write_json(args.output/"status.json", {"stage": "test", "run": str(checkpoint)})
                    metrics = evaluate(path, partition, env, checkpoint, "test", args)
                    results.append({"kind": kind, "weight": weight, "seed": seed,
                                    "selected": weight == selected[kind], "fixed": weight == FIXED[kind], **metrics})
                    write_json(out/"test.json", results)
    write_json(args.output/"status.json", {"stage": "complete"})
    print("BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
