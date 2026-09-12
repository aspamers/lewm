"""Check archive schema and alignment between recorded states and simulator pixels."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image
import stable_worldmodel  # noqa: F401
import torch


def check(path, env, output):
    output.mkdir(parents=True, exist_ok=True)
    env_id = {"tworoom": "swm/TwoRoom-v1", "pusht": "swm/PushT-v1"}[env]
    state_key = "proprio" if env == "tworoom" else "state"
    with h5py.File(path, "r") as f:
        required = {"pixels", "action", "ep_len", "ep_offset", state_key}
        if not required.issubset(f.keys()):
            raise ValueError(f"Missing columns: {required-set(f.keys())}")
        index = int(f["ep_offset"][0])
        pixel = f["pixels"][index]
        if pixel.shape[0] == 3:
            pixel = pixel.transpose(1, 2, 0)
        recorded = Image.fromarray(pixel)
        simulator = gym.make(env_id, render_mode="rgb_array")
        seed = int(np.asarray(f["seed"][index]).item()) if "seed" in f else 6173
        simulator.reset(seed=seed)
        simulator.unwrapped._set_state(f[state_key][index])
        rendered = Image.fromarray(simulator.render()).resize(recorded.size, Image.Resampling.BILINEAR)
        next_observation, _, _, _, _ = simulator.step(f["action"][index])
        next_state = (simulator.unwrapped.agent_position.numpy() if env == "tworoom"
                      else np.asarray(next_observation["state"]))
        transition_error = float(np.max(np.abs(next_state-f[state_key][index+1])))
        simulator.close()
        recorded.save(output/"recorded.png")
        rendered.save(output/"reconstructed.png")
        mse = float(np.mean(((np.asarray(recorded).astype(float)-np.asarray(rendered))/255)**2))
        report = {"environment": env, "path": str(path), "episodes": len(f["ep_len"]),
                  "frames": len(f["pixels"]), "action_shape": list(f["action"].shape),
                  "pixel_shape": list(f["pixels"].shape), "pixel_mse": mse,
                  "one_step_state_max_abs_error": transition_error,
                  "generated_evaluation_seed": "seed" not in f,
                  "columns": list(f.keys())}
    (output/"data-check.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", choices=["tworoom", "pusht"])
    parser.add_argument("path", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(check(args.path, args.environment, args.output), indent=2))
