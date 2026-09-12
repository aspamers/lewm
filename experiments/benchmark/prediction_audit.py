"""Compare old/new checkpoints under isolated normalization/dropout changes."""
import hashlib
import json
from pathlib import Path
import statistics
import torch
from torch.utils.data import DataLoader
from run import Clips, write_json
from runtime import build_model, normalize_pixels


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path("outputs/prediction-audit")
    output.mkdir(exist_ok=True)
    partition = json.loads(Path("outputs/two-environment-benchmark/tworoom/partition.json").read_text())
    path = Path("/data/datasets/tworoom_frames.h5")
    mean,std = (torch.tensor(partition[k],device="cuda") for k in ("action_mean","action_std"))
    def data(split, seed, steps):
        return [{"pixels":normalize_pixels(b["pixels"].cuda()),
                 "action":torch.nan_to_num((b["action"].cuda()-mean)/std).reshape(64,4,-1)}
                for b in DataLoader(Clips(path,partition[split],seed,steps,64),batch_size=64)]
    validation = data("validation",7791,4)
    calibration = data("train",8821,8)
    checkpoints = [
        "two-environment-benchmark/tworoom/gaussian-w0.03-s31",
        "two-environment-benchmark/tworoom/variance_decorrelation-w0.1-s31",
        "scale-fix-trial/legacy-fp32", "scale-fix-trial/corrected-fp32",
        "floor-strength/floor0.3-s31", "floor-strength/floor1-s31"]
    results = []
    for checkpoint in checkpoints:
        model = build_model(2).cuda()
        weight_path = Path("outputs")/checkpoint/"weights.pt"
        state = torch.load(weight_path,map_location="cuda",weights_only=True)
        for mode in ("eval", "predictor_bn_batch", "encoder_bn_batch", "both_bn_batch",
                     "predictor_dropout_on", "both_bn_and_dropout", "eval_bf16", "calibrated_predictor_bn",
                     "calibrated_encoder_bn", "calibrated_both_bn"):
            model.load_state_dict(state)
            model.eval()
            torch.manual_seed(9201)
            if mode in ("encoder_bn_batch","both_bn_batch","both_bn_and_dropout"):
                model.projector.net[1].train()
            if mode in ("predictor_bn_batch","both_bn_batch","both_bn_and_dropout"):
                model.pred_proj.net[1].train()
            if mode in ("predictor_dropout_on","both_bn_and_dropout"):
                # Also enables functional attention dropout, not only nn.Dropout.
                model.predictor.train()
            if mode in ("calibrated_encoder_bn", "calibrated_both_bn"):
                collected = []
                hook = model.projector.net[1].register_forward_pre_hook(
                    lambda mod,args: collected.append(args[0].detach().double().cpu()))
                for batch in calibration:
                    model.encode(dict(batch))
                hook.remove()
                values = torch.cat(collected)
                bn = model.projector.net[1]
                bn.running_mean.copy_(values.mean(0))
                bn.running_var.copy_(values.var(0,unbiased=True))
            if mode in ("calibrated_predictor_bn", "calibrated_both_bn"):
                # Population moments at the actual inference-time predictor
                # input, estimated from training clips only. No weights change.
                collected = []
                hook = model.pred_proj.net[1].register_forward_pre_hook(
                    lambda mod,args: collected.append(args[0].detach().double().cpu()))
                for batch in calibration:
                    enc = model.encode(dict(batch))
                    model.predict(enc["emb"][:,:3],enc["act_emb"][:,:3])
                hook.remove()
                values = torch.cat(collected)
                bn = model.pred_proj.net[1]
                bn.running_mean.copy_(values.mean(0))
                bn.running_var.copy_(values.var(0,unbiased=True))
            measures = []
            for batch in validation:
                captures = {}
                def capture(mod,args):
                    x = args[0].float()
                    captures["pred_bn_mean_shift_rms"] = float(((x.mean(0)-mod.running_mean)/
                                                              (mod.running_var+mod.eps).sqrt()).square().mean().sqrt())
                    captures["pred_bn_variance_ratio"] = float((x.var(0,unbiased=False)/(mod.running_var+mod.eps)).mean())
                hook = model.pred_proj.net[1].register_forward_pre_hook(capture)
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=mode=="eval_bf16"):
                    enc = model.encode(dict(batch))
                    z = enc["emb"].float()
                    p = model.predict(enc["emb"][:,:3],enc["act_emb"][:,:3]).float()
                hook.remove()
                target = z[:,1:]
                residual = p-target
                variance = target.var((0,1),unbiased=False).mean().clamp_min(1e-20)
                measures.append({"std":float(z.std(0).mean()), "mse":float(residual.square().mean()),
                                 "normalized_mse":float(residual.square().mean()/variance),
                                 "centered_residual_mse":float((residual-residual.mean((0,1),keepdim=True)).square().mean()),
                                 "persistence_normalized_mse":float((z[:,:3]-target).square().mean()/variance),
                                 **captures})
            row = {"checkpoint":checkpoint,"mode":mode,
                   **{k:statistics.mean(m[k] for m in measures) for k in measures[0]}}
            results.append(row)
            print(row,flush=True)
            write_json(output/"results.json",results)
        del model,state
        torch.cuda.empty_cache()
    write_json(output/"manifest.json",{"calibration_split":"train","calibration_clips":512,
                                      "validation_clips":256,"validation_seed":7791,
                                      "checkpoints":{p:hashlib.sha256((Path("outputs")/p/"weights.pt").read_bytes()).hexdigest()
                                                     for p in checkpoints},
                                      "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(output/"status.json",{"stage":"complete"})


if __name__=="__main__":
    main()
