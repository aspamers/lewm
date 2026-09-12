"""Lossless per-frame HDF5 training cache for the archive's 100-frame chunks."""
import argparse
import json
from pathlib import Path
import shutil
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def convert(source, destination):
    marker = destination.with_suffix(".cache.json")
    if destination.exists() and marker.exists():
        return
    temporary = destination.with_suffix(".part")
    status = destination.with_suffix(".progress.json")
    start_time = time.perf_counter()
    with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
        for key in ("ep_len", "ep_offset", "action"):
            src.copy(key, dst)
        pixels = src["pixels"]
        n = len(pixels)
        out = dst.create_dataset("pixels", shape=pixels.shape, dtype=pixels.dtype,
                                 chunks=(1, *pixels.shape[1:]), compression="lzf", shuffle=True)
        block_size = pixels.chunks[0] if pixels.chunks else 100
        for start in range(0, n, block_size):
            if shutil.disk_usage(destination.parent).free < 30*1024**3:
                raise RuntimeError("Preserving 30 GiB free-disk reserve")
            out[start:start+block_size] = pixels[start:start+block_size]
            if (start//block_size) % 100 == 0:
                status.write_text(json.dumps({"frames": start, "total": n,
                                              "seconds": time.perf_counter()-start_time}))
        indices = np.unique(np.linspace(0, n-1, 41, dtype=int))
        for index in indices:
            if not np.array_equal(out[index], pixels[index]):
                raise RuntimeError("Lossless cache verification failed")
        for key in ("ep_len", "ep_offset", "action"):
            if not np.array_equal(src[key][:], dst[key][:], equal_nan=True):
                raise RuntimeError("Metadata cache verification failed")
        record = {"source": str(source), "source_bytes": source.stat().st_size,
                  "source_mtime_ns": source.stat().st_mtime_ns, "frames": n,
                  "shape": list(pixels.shape), "compression": "lzf", "chunk_frames": 1,
                  "verified_indices": indices.tolist(), "seconds": time.perf_counter()-start_time}
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    marker.write_text(json.dumps(record, indent=2))
    status.write_text(json.dumps({"stage": "ready", **record}))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
