import importlib.util
from pathlib import Path

import h5py
import numpy as np

spec = importlib.util.spec_from_file_location("rechunk", Path(__file__).parents[1]/"experiments/benchmark/rechunk.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_lossless_cache_preserves_pixels_actions_and_episode_boundaries(tmp_path):
    source, target = tmp_path/"source.h5", tmp_path/"frames.h5"
    pixels = np.random.default_rng(14).integers(0, 256, (17, 8, 8, 3), dtype=np.uint8)
    actions = np.arange(34, dtype=float).reshape(17, 2)
    actions[-1] = np.nan
    with h5py.File(source, "w") as f:
        f.create_dataset("pixels", data=pixels, chunks=(5, 8, 8, 3))
        f["action"] = actions
        f["ep_len"] = [8, 9]
        f["ep_offset"] = [0, 8]
    module.convert(source, target)
    with h5py.File(target, "r") as f:
        assert f["pixels"].chunks == (1, 8, 8, 3)
        np.testing.assert_array_equal(f["pixels"][:], pixels)
        np.testing.assert_array_equal(f["action"][:], actions)
        np.testing.assert_array_equal(f["ep_len"][:], [8, 9])
        np.testing.assert_array_equal(f["ep_offset"][:], [0, 8])
    assert target.with_suffix(".cache.json").exists()
