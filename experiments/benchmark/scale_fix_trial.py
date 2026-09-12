"""Paired 2x2 loss/precision diagnosis on TwoRoom; no test-goal selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from lewm.regularizers import VarianceFloorRegularizer
from run import Clips, write_json
from runtime import build_model, normalize_pixels, objective


class Legacy(torch.nn.Module):
    def forward(self, z):
        z = z.float()
        c = z-z.mean(1, keepdim=True)
        v = c.square().sum(1)/(z.shape[1]-1)
        floor = torch.relu(1-(v+1e-4).sqrt()).square().mean()
        q = c/v.clamp_min(1e-4).sqrt().unsqueeze(1)
        corr = q.transpose(-1,-2)@q/(z.shape[1]-1)
        off = corr-torch.diag_embed(corr.diagonal(dim1=-2,dim2=-1))
        d = z.shape[-1]
        return floor+off.square().sum((-1,-2)).mean()/(d*(d-1))


@torch.no_grad()
def spread(model, batch):
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    results = {}
    for batch_stats in (False, True):
        for amp in (False, True):
            for k,v in model.named_buffers():
                v.copy_(buffers[k])
            model.eval()
            if batch_stats:
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.train()
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=amp):
                z = model.encode({"pixels":batch})["emb"]
            results[f"{'batch' if batch_stats else 'eval'}_{'bf16' if amp else 'fp32'}"] = float(z.float().std(0).mean())
    for k,v in model.named_buffers():
        v.copy_(buffers[k])
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("outputs/scale-fix-trial"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    partition = json.loads(Path("outputs/two-environment-benchmark/tworoom/partition.json").read_text())
    path = Path("/data/datasets/tworoom_frames.h5")
    probe = next(iter(DataLoader(Clips(path,partition["validation"],7781,1,64), batch_size=64)))
    probe = normalize_pixels(probe["pixels"].cuda())
    manifest = {"steps":args.steps,"seed":31,"batch":64,"weight":.1,
                "encoder_gradient_checkpointing":True,"tf32":False,
                "sources":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__),Path("src/lewm/regularizers.py"),Path("experiments/benchmark/runtime.py")]}}
    if (args.output/"manifest.json").exists() and json.loads((args.output/"manifest.json").read_text()) != manifest:
        raise RuntimeError("Use a fresh directory for changed settings/code")
    write_json(args.output/"manifest.json",manifest)
    results = []
    for corrected, amp in ((False,True),(True,True),(False,False),(True,False)):
        name = f"{'corrected' if corrected else 'legacy'}-{'bf16' if amp else 'fp32'}"
        out = args.output/name
        out.mkdir(exist_ok=True)
        torch.manual_seed(31)
        model = build_model(2).cuda().train()
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        regularizer = VarianceFloorRegularizer(True) if corrected else Legacy()
        optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-3)
        loader = DataLoader(Clips(path,partition["train"],531,args.steps,64),batch_size=64,num_workers=2,
                            pin_memory=True,generator=torch.Generator().manual_seed(731))
        mean = torch.tensor(partition["action_mean"],device="cuda")
        std = torch.tensor(partition["action_std"],device="cuda")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        curve = []
        for step,batch in enumerate(loader,1):
            pixels = normalize_pixels(batch["pixels"].cuda())
            actions = torch.nan_to_num((batch["action"].cuda()-mean)/std).reshape(64,4,-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=amp):
                loss,pred,reg = objective(model,{"pixels":pixels,"action":actions},regularizer,.1,3100000+step)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {name} at {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step%50==0 or step==args.steps:
                row = {"step":step,"prediction":float(pred),"regularizer":float(reg),
                       "spread":spread(model,probe),"seconds":time.perf_counter()-start}
                curve.append(row)
                write_json(out/"curve.json",curve)
                write_json(args.output/"status.json",{"stage":"training","arm":name,**row})
                print(name,row,flush=True)
        torch.save(model.state_dict(),out/"weights.pt")
        results.append({"arm":name,**curve[-1],"peak_bytes":torch.cuda.max_memory_allocated()})
        write_json(args.output/"results.json",results)
        del model,optimizer,loader,loss,pred,reg
        torch.cuda.empty_cache()
    write_json(args.output/"status.json",{"stage":"complete"})


if __name__ == "__main__":
    main()
