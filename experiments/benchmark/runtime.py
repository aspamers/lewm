"""Shared full LeWM architecture and preprocessing for the real-environment pilot."""
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch


def build_model(action_dim):
    config = OmegaConf.load(Path(__file__).resolve().parents[2] / "config/train/model/lewm.yaml")
    config = OmegaConf.merge({"img_size": 224, "embed_dim": 192, "history_size": 3}, {"model": config})
    config.model.action_encoder.input_dim = 5*action_dim
    return hydra.utils.instantiate(config.model)


def normalize_pixels(pixels):
    # Both HDF5 clips and policy images are channel-first before this function.
    pixels = pixels.float()/255
    if pixels.shape[-2:] != (224, 224):
        shape = pixels.shape
        pixels = torch.nn.functional.interpolate(pixels.reshape(-1, *shape[-3:]),
                   size=(224, 224), mode="bilinear", align_corners=False, antialias=True).reshape(*shape[:-2], 224, 224)
    mean = pixels.new_tensor([.485, .456, .406])[:, None, None]
    std = pixels.new_tensor([.229, .224, .225])[:, None, None]
    return (pixels-mean)/std


def objective(model, batch, regularizer, weight, reg_seed):
    out = model.encode(dict(batch))
    pred = model.predict(out["emb"][:, :3], out["act_emb"][:, :3])
    prediction = (pred.float()-out["emb"][:, 1:4].float()).square().mean()
    # Full predictor has dropout. Isolate projection RNG so it cannot change
    # later dropout masks differently between regularizers.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(reg_seed)
        penalty = regularizer(out["emb"].transpose(0, 1))
    return prediction+weight*penalty, prediction, penalty
