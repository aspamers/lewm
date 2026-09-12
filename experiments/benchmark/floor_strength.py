"""Independent floor-weight sweep in full precision, with validation diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

import torch
from torch.utils.data import DataLoader
from lewm.regularizers import VarianceFloorRegularizer
from run import Clips, write_json
from runtime import build_model, normalize_pixels, objective


@torch.no_grad()
def measure(model, batches):
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    model.eval()
    values = []
    floor = VarianceFloorRegularizer(False)
    corr = VarianceFloorRegularizer(True, floor_weight=0.)
    for batch in batches:
        encoded = model.encode(dict(batch))
        z, actions = encoded["emb"], encoded["act_emb"]
        target = z[:,1:]
        prediction = model.predict(z[:,:3],actions[:,:3])
        shuffled = model.predict(z[:,:3],actions.roll(1,0)[:,:3])
        mse = (prediction-target).square().mean()
        persistence = (z[:,:3]-target).square().mean()
        variance = target.var(dim=(0,1),unbiased=False).mean().clamp_min(1e-20)
        values.append({"eval_std":float(z.std(0).mean()),"floor_penalty":float(floor(z.transpose(0,1))),
                       "correlation_penalty":float(corr(z.transpose(0,1))),
                       "prediction_mse":float(mse),"normalized_mse":float(mse/variance),
                       "persistence_normalized_mse":float(persistence/variance),
                       "shuffled_action_normalized_mse":float((shuffled-target).square().mean()/variance)})
    result = {key:statistics.mean(v[key] for v in values) for key in values[0]}
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            module.train()
    batch_std = []
    for batch in batches:
        z = model.encode({"pixels":batch["pixels"]})["emb"]
        batch_std.append(float(z.std(0).mean()))
    result["batch_std"] = statistics.mean(batch_std)
    for key,value in model.named_buffers():
        value.copy_(buffers[key])
    model.train()
    return result


def report(output, results, settings):
    lines = ["# Independent variance-floor strength experiment", "",
             f"TwoRoom, {settings['steps']} updates, batch 64, full float32 with TF32 disabled. "
             f"Seeds {settings['seeds']}; correlation coefficient fixed at 0.1.",
             "Architecture, optimizer, initialization and minibatches are paired across floor strengths.",
             "Encoder activation checkpointing is enabled for all arms. Four fixed validation batches",
             "provide diagnostics; no planning test goals or new coefficient selection are used.", "",
             "| Floor | Seed | Batch std | Eval std | Normalized prediction MSE | Persistence MSE (normalized) | Shuffled-action MSE (normalized) |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        v = r["validation"]
        lines.append(f"| {r['floor_weight']:g} | {r['seed']} | {v['batch_std']:.4f} | {v['eval_std']:.4f} | "
                     f"{v['normalized_mse']:.4f} | {v['persistence_normalized_mse']:.4f} | {v['shuffled_action_normalized_mse']:.4f} |")
    lines += ["", "A larger standard deviation by itself is not proof of a better representation.",
              "Normalized errors divide by target-embedding variance, making uniform scale changes cancel.",
              "Persistence predicts the next embedding equals the current embedding. Shuffled actions",
              "are a deterministic batch permutation; increased error suggests action sensitivity, not",
              "necessarily correct causal dynamics. These checks do not replace physical-state probes",
              "or planning evaluation. Two seeds and 250 updates remain exploratory.", ""]
    (output/"REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=250)
    parser.add_argument("--seeds",type=int,nargs="+",default=[31,32])
    parser.add_argument("--weights",type=float,nargs="+",default=[.1,.3,1.,3.])
    parser.add_argument("--output",type=Path,default=Path("outputs/floor-strength"))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    partition_path = Path("outputs/two-environment-benchmark/tworoom/partition.json")
    partition = json.loads(partition_path.read_text())
    path = Path("/data/datasets/tworoom_frames.h5")
    mean = torch.tensor(partition["action_mean"],device="cuda")
    std = torch.tensor(partition["action_std"],device="cuda")
    probes = []
    for batch in DataLoader(Clips(path,partition["validation"],7791,4,64),batch_size=64):
        probes.append({"pixels":normalize_pixels(batch["pixels"].cuda()),
                       "action":torch.nan_to_num((batch["action"].cuda()-mean)/std).reshape(64,4,-1)})
    settings = {"steps":args.steps,"seeds":args.seeds,"floor_weights":args.weights,
                "correlation_weight":.1,"precision":"float32","batch":64,"probe_seed":7791,
                "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
                "sources":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__),Path("src/lewm/regularizers.py"),Path("src/lewm/jepa.py"),
                            Path("src/lewm/module.py"),Path("experiments/benchmark/run.py"),
                            Path("experiments/benchmark/runtime.py"),Path("config/train/model/lewm.yaml"),partition_path]}}
    if (args.output/"manifest.json").exists() and json.loads((args.output/"manifest.json").read_text())!=settings:
        raise RuntimeError("Use a fresh output directory for changed settings/source")
    write_json(args.output/"manifest.json",settings)
    results = []
    for seed in args.seeds:
        for weight in args.weights:
            out = args.output/f"floor{weight:g}-s{seed}"
            out.mkdir(exist_ok=True)
            if (out/"result.json").exists():
                results.append(json.loads((out/"result.json").read_text()))
                continue
            torch.manual_seed(seed)
            model = build_model(2).cuda().train()
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            regularizer = VarianceFloorRegularizer(True,floor_weight=weight,correlation_weight=.1)
            optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-3)
            loader = DataLoader(Clips(path,partition["train"],seed+500,args.steps,64),batch_size=64,
                                num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(seed+700))
            curve = []
            start = time.perf_counter()
            for step,batch in enumerate(loader,1):
                x = normalize_pixels(batch["pixels"].cuda())
                a = torch.nan_to_num((batch["action"].cuda()-mean)/std).reshape(64,4,-1)
                optimizer.zero_grad(set_to_none=True)
                loss,pred,reg = objective(model,{"pixels":x,"action":a},regularizer,1.,seed*100000+step)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss: {out}, {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step%50==0 or step==args.steps:
                    row = {"step":step,"prediction_loss":float(pred),"weighted_regularizer":float(reg),
                           "validation":measure(model,probes),"seconds":time.perf_counter()-start}
                    curve.append(row)
                    write_json(out/"curve.json",curve)
                    write_json(args.output/"status.json",{"stage":"training","run":str(out),**row})
                    print(str(out),row,flush=True)
            torch.save(model.state_dict(),out/"weights.pt")
            result = {"floor_weight":weight,"correlation_weight":.1,"seed":seed,**curve[-1]}
            write_json(out/"result.json",result)
            results.append(result)
            write_json(args.output/"results.json",results)
            del model,optimizer,loader,loss,pred,reg
            torch.cuda.empty_cache()
    report(args.output,results,settings)
    if args.steps==250 and args.seeds==[31,32] and args.weights==[.1,.3,1.,3.]:
        dest = Path("experiments/results/floor-strength")
        dest.mkdir(parents=True,exist_ok=True)
        for filename in ("REPORT.md","results.json","manifest.json"):
            shutil.copy2(args.output/filename,dest/filename)
        for out in args.output.iterdir():
            if out.is_dir() and (out/"curve.json").exists():
                shutil.copy2(out/"curve.json",dest/(out.name+"-curve.json"))
    write_json(args.output/"status.json",{"stage":"complete"})


if __name__=="__main__":
    main()
