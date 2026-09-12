"""Alternating-order GPU forward/backward timing of the two regularizers."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from lewm.regularizers import make_regularizer


def measure(output):
    torch.set_num_threads(4)
    torch.manual_seed(904)
    z = torch.randn(4, 64, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kinds = ["gaussian", "variance_decorrelation"]
    regs = {k: make_regularizer(k, num_proj=1024).cuda() for k in kinds}
    for i in range(40):
        for kind in kinds:
            z.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                regs[kind](z).backward()
    torch.cuda.synchronize()
    measurements = {k: [] for k in kinds}
    wall = {k: [] for k in kinds}
    for i in range(200):
        for kind in (kinds if i%2 == 0 else kinds[::-1]):
            z.grad = None
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                regs[kind](z).backward()
            end.record()
            end.synchronize()
            measurements[kind].append(begin.elapsed_time(end))
            wall[kind].append(1000*(time.perf_counter()-start))
    rows = {k: {"median_gpu_ms": float(np.median(measurements[k])),
                "p95_gpu_ms": float(np.percentile(measurements[k], 95)),
                "median_wall_ms": float(np.median(wall[k]))} for k in kinds}
    output.write_text(json.dumps({"shape": list(z.shape), "dtype": str(z.dtype),
                                 "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                                 "warmup": 40, "repeats": 200, "results": rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    measure(parser.parse_args().output)
