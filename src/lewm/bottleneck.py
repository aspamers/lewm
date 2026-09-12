"""Observation-grounded, open-loop predictive bottleneck experiment."""

import torch
from torch import nn
from torch.nn import functional as F

from lewm.jepa import JEPA
from lewm.module import ARPredictor


class BottleneckPredictor(nn.Module):
    """Vary latent width while holding transformer and action capacity fixed."""

    def __init__(self, latent_dim, hidden_dim=192, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.core = ARPredictor(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                output_dim=hidden_dim, **kwargs)

    def forward(self, x, c, attn_mask=None):
        return self.core(self.input_proj(x), c, attn_mask=attn_mask)


class ObservationDecoder(nn.Module):
    """Small decoder with unconstrained output for ImageNet-normalized targets."""

    def __init__(self, latent_dim, image_size=224, hidden_dim=64):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.project = nn.Linear(latent_dim, hidden_dim * 7 * 7)
        self.convs = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
        )

    def forward(self, z):
        x = self.project(z).reshape(-1, self.hidden_dim, 7, 7)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        return self.convs(x)


def multiscale_error(pred, target, scales=(1, 2, 4)):
    """Per-image MSE, equally weighted across fixed spatial resolutions."""
    errors = []
    for scale in scales:
        p = F.avg_pool2d(pred, scale) if scale > 1 else pred
        t = F.avg_pool2d(target, scale) if scale > 1 else target
        errors.append((p - t).square().mean(dim=(-3, -2, -1)))
    return torch.stack(errors).mean(0)


class PredictiveBottleneck(JEPA):
    def __init__(self, decoder=None, planning_cost="latent", **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder
        self.planning_cost = planning_cost

    def criterion(self, info_dict):
        if self.planning_cost == "decoded":
            return self.decoded_terminal_cost(info_dict["predicted_emb"], info_dict["goal"])
        if self.planning_cost != "latent":
            raise ValueError(f"Unknown planning cost: {self.planning_cost}")
        return super().criterion(info_dict)

    def decoded_terminal_cost(self, predicted_emb, goal_pixels):
        """Diagnostic cost to the actual preprocessed goal image, never D(E(goal))."""
        if self.decoder is None:
            raise ValueError("Decoded cost requires an observation decoder")
        b, s = predicted_emb.shape[:2]
        decoded = self.decoder(predicted_emb[:, :, -1].reshape(b * s, -1))
        target = goal_pixels[:, :, -1].expand(b, s, *decoded.shape[-3:])
        return multiscale_error(decoded, target.reshape_as(decoded)).reshape(b, s)


def predictive_loss(model, batch, sigreg, *, history_size=3, horizons=(1, 3, 5),
                    latent_weight=0.1, observation_weight=1.0, sigreg_weight=0.0):
    """Roll out from history only; future frames appear solely as loss targets.

    action[t] causes pixels[t] -> pixels[t+1]. Horizon k targets H-1+k.
    Baseline uses the original teacher-forced one-step latent loss; observation
    variants add open-loop future image supervision, isolating that intervention.
    """
    if not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be distinct positive integers")
    if history_size < 1:
        raise ValueError("history_size must be positive")
    if min(latent_weight, observation_weight, sigreg_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    required = history_size + max(horizons)
    if batch["pixels"].shape[1] < required or batch["action"].shape[1] < required - 1:
        raise ValueError(f"Need {required} frames and {required-1} aligned actions")
    if observation_weight and model.decoder is None:
        raise ValueError("Observation loss requires a decoder")
    if not torch.isfinite(batch["action"][:, :required-1]).all():
        raise ValueError("Invalid actions: exclude windows crossing episode boundaries")
    output = model.encode(dict(batch))
    z, a = output["emb"], output["act_emb"]
    one_step = model.predict(z[:, :history_size], a[:, :history_size])
    latent = F.mse_loss(one_step, z[:, 1:history_size+1])
    observation = z.new_zeros(())
    metrics = {"pred_loss": latent}
    if observation_weight:
        context = z[:, :history_size]
        for k in range(1, max(horizons) + 1):
            start = k - 1
            pred = model.predict(context[:, -history_size:], a[:, start:start+history_size])[:, -1:]
            context = torch.cat((context, pred), dim=1)
            if k in horizons:
                decoded = model.decoder(pred[:, 0])
                error = multiscale_error(decoded, batch["pixels"][:, history_size-1+k]).mean()
                metrics[f"obs_h{k}_loss"] = error
                observation = observation + error / len(horizons)
    regularizer = sigreg(z.transpose(0, 1)) if sigreg_weight else z.new_zeros(())
    metrics.update(obs_loss=observation, sigreg_loss=regularizer,
                   loss=latent_weight * latent + observation_weight * observation + sigreg_weight * regularizer)
    return metrics
