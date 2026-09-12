"""Read-only checkpoint audit of scale, precision and BatchNorm behavior."""
import json
from pathlib import Path
import statistics
import torch
from torch.utils.data import DataLoader
from run import Clips, write_json
from runtime import build_model, normalize_pixels
from lewm.regularizers import VarianceFloorRegularizer


def stats(z):
    z = z.float().transpose(0, 1)
    c = z-z.mean(1, keepdim=True)
    v = c.square().sum(1)/(z.shape[1]-1)
    q = c/v.clamp_min(1e-4).sqrt().unsqueeze(1)
    r = q.transpose(-1, -2)@q/(z.shape[1]-1)
    d = z.shape[-1]
    off = r-torch.diag_embed(r.diagonal(dim1=-2, dim2=-1))
    return {"std": float(v.sqrt().mean()), "floor": float(torch.relu(1-(v+1e-4).sqrt()).square().mean()),
            "correlation_penalty": float(off.square().sum((-1,-2)).mean()/(d*(d-1))),
            "fraction_below_correlation_epsilon": float((v<1e-4).float().mean())}


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    destination = Path("outputs/scale-audit")
    destination.mkdir(exist_ok=True)
    partition = json.loads(Path("outputs/two-environment-benchmark/tworoom/partition.json").read_text())
    batches = {}
    for split in ("train", "validation"):
        clips = Clips(Path("/data/datasets/tworoom_frames.h5"), partition[split], 4491, 4, 64)
        batches[split] = list(DataLoader(clips, batch_size=64, num_workers=0))
    rows = []
    for root in ("two-environment-benchmark", "two-environment-1000"):
        for seed in (31, 32):
            for kind, weight in (("gaussian", .03), ("variance_decorrelation", .1)):
                checkpoint = Path("outputs")/root/"tworoom"/f"{kind}-w{weight}-s{seed}"/"weights.pt"
                model = build_model(2).cuda()
                state = torch.load(checkpoint, map_location="cuda", weights_only=True)
                for mode in ("eval", "batchnorm_train"):
                    for precision in ("float32", "bfloat16"):
                        for split in batches:
                            model.load_state_dict(state)
                            model.eval()
                            if mode == "batchnorm_train":
                                for m in model.modules():
                                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                                        m.train()
                            captures = {}
                            hooks = []
                            for name, module in (("encoder", model.projector.net[0]), ("pre_bn", model.projector.net[1])):
                                def save(mod, args, key=name):
                                    captures[key] = args[0].detach().float().std(0).mean().item()
                                hooks.append(module.register_forward_pre_hook(save))
                            measures = []
                            for batch in batches[split]:
                                x = normalize_pixels(batch["pixels"].cuda())
                                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision=="bfloat16"):
                                    z = model.encode({"pixels": x})["emb"]
                                measures.append({**stats(z), **captures})
                            for hook in hooks:
                                hook.remove()
                            row = {"checkpoint": str(checkpoint), "mode": mode, "precision": precision, "split": split,
                                   **{key: statistics.mean(m[key] for m in measures) for key in measures[0]}}
                            rows.append(row)
                del model, state
                torch.cuda.empty_cache()
                write_json(destination/"results.json", rows)
                print(str(checkpoint), "audited", flush=True)
    print("SCALE AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
