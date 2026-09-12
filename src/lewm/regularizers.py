"""Distribution-target ablations with fixed analytic characteristic functions.

The mixture is 0.5 N(+m, s I) + 0.5 N(-m, s I), m_i=sqrt(r), s=1-r.
Each coordinate has mean zero and variance one. Covariance is s I + m m^T,
NOT identity. A covariance-matched Gaussian separates this from multimodality.
"""
import math

import torch
from torch import nn

from lewm.module import SIGReg


class DistributionRegularizer(SIGReg):
    def __init__(self, kind="gaussian", separation=0.75, **kwargs):
        super().__init__(**kwargs)
        if kind not in {"gaussian", "mixture", "matched_gaussian"}:
            raise ValueError(f"Unknown target: {kind}")
        if not 0 <= separation < 1:
            raise ValueError("separation must be in [0, 1)")
        self.kind = kind
        self.separation = separation

    def target_cf(self, directions):
        """Real CF for unit projection directions (D, P), returned as (P, K).

        All targets are centrally symmetric, so their imaginary CF is zero.
        The SAME multivariate mixture supplies every projected target.
        """
        if self.kind == "gaussian":
            return self.phi.expand(directions.shape[1], -1)
        projected_mean = math.sqrt(self.separation) * directions.sum(0)
        t = self.t
        if self.kind == "matched_gaussian":
            variance = 1 - self.separation + projected_mean.square()
            return torch.exp(-0.5 * variance[:, None] * t.square())
        return (torch.exp(-0.5 * (1-self.separation) * t.square())
                * torch.cos(projected_mean[:, None] * t))

    def forward(self, z):
        if z.ndim != 3:
            raise ValueError("Expected (time, batch, latent)")
        # Keep the estimator, frequency grid, weighting and random projections
        # identical to upstream SIGReg; change only the target characteristic function.
        directions = torch.randn(z.shape[-1], self.num_proj, device=z.device)
        directions = directions / directions.norm(dim=0)
        phase = (z.float() @ directions).unsqueeze(-1) * self.t
        error = ((phase.cos().mean(-3) - self.target_cf(directions)).square()
                 + phase.sin().mean(-3).square())
        return ((error @ self.weights) * z.shape[-2]).mean()


class MomentRegularizer(nn.Module):
    """Mean anchor, per-coordinate variance floor, off-diagonal covariance.

    No Gaussianity or modality target. Loss scales differ from CF matching;
    compare a coefficient sweep, not just equal numeric coefficients.
    """
    def forward(self, z):
        if z.shape[1] < 2:
            raise ValueError("Moment regularization needs at least two samples")
        z = z.float()
        mean = z.mean(1, keepdim=True)
        centered = z - mean
        cov = centered.transpose(-1, -2) @ centered / (z.shape[1] - 1)
        var = cov.diagonal(dim1=-2, dim2=-1)
        floor = torch.relu(1 - (var + 1e-4).sqrt()).square().mean()
        off_diagonal = cov - torch.diag_embed(var)
        return mean.square().mean() + floor + off_diagonal.square().sum((-1, -2)).mean() / z.shape[-1]


class NoRegularizer(nn.Module):
    def forward(self, z):
        return z.new_zeros(())


def make_regularizer(kind, num_proj=64, separation=.75, knots=17):
    if kind == "none":
        return NoRegularizer()
    if kind == "moments":
        return MomentRegularizer()
    return DistributionRegularizer(kind, separation=separation, num_proj=num_proj, knots=knots)
