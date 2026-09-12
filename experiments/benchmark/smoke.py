"""Simulator rendering and full-model throughput checks; not scientific results."""
import argparse
import gc
import json
from pathlib import Path
import time

import gymnasium as gym
import stable_worldmodel  # noqa: F401
import torch
from PIL import Image

from runtime import build_model, normalize_pixels, objective
from lewm.regularizers import make_regularizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    report = {"gpu": torch.cuda.get_device_name(), "environments": {}, "throughput": []}
    for name, env_id in [("tworoom", "swm/TwoRoom-v1"), ("pusht", "swm/PushT-v1")]:
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=91)
        env.step(env.action_space.sample())
        frame = env.render()
        Image.fromarray(frame).save(args.output / (name + ".png"))
        report["environments"][name] = {"action_shape": env.action_space.shape, "render_shape": frame.shape}
        env.close()
    for batch_size in [8, 16, 32, 64]:
        try:
            torch.manual_seed(51)
            model = build_model(2).cuda().train()
            reg = make_regularizer("gaussian", num_proj=1024).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            batch = {"pixels": normalize_pixels(torch.randint(256, (batch_size, 4, 3, 224, 224), device="cuda")),
                     "action": torch.randn(batch_size, 4, 10, device="cuda")}
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for step in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _, _ = objective(model, batch, reg, .09, 91+step)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                torch.cuda.synchronize()
                timings.append(time.perf_counter()-start)
            row = {"batch": batch_size, "seconds_per_step": sum(timings[2:])/3,
                   "peak_bytes": torch.cuda.max_memory_allocated(),
                   "parameters": sum(p.numel() for p in model.parameters())}
            report["throughput"].append(row)
            print(row, flush=True)
        except torch.cuda.OutOfMemoryError:
            report["throughput"].append({"batch": batch_size, "error": "CUDA out of memory"})
            break
        finally:
            for name in ("model", "reg", "optimizer", "batch", "loss"):
                globals().pop(name, None)
            gc.collect()
            torch.cuda.empty_cache()
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
