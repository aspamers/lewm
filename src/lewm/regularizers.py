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


class StudentTRegularizer(DistributionRegularizer):
    """Spherical multivariate Student-t, mean zero and covariance I.

    X = G sqrt((nu-2)/U), G~N(0,I), U~chi-square(nu), one shared U per vector.
    Every unit projection has the same variance-one univariate Student-t law.
    For nu=3,5,9 its characteristic function has an exact polynomial form,
    avoiding special-function dependencies or sampling noise in the target.
    SIGReg's Gaussian integration window is intentionally left unchanged.
    """
    def __init__(self, df=5, **kwargs):
        if df not in (3, 5, 9):
            raise ValueError("Supported degrees of freedom are 3, 5 and 9")
        super().__init__(kind="gaussian", **kwargs)
        self.df = df
        self.kind = "student_t"
        x = math.sqrt(df - 2) * self.t.abs()
        if df == 3:
            polynomial = 1 + x
        elif df == 5:
            polynomial = 1 + x + x.square() / 3
        else:
            polynomial = 1 + x + 3*x.square()/7 + 2*x.pow(3)/21 + x.pow(4)/105
        self.register_buffer("student_phi", polynomial * torch.exp(-x))

    def target_cf(self, directions):
        return self.student_phi.expand(directions.shape[1], -1)


class VarianceFloorRegularizer(nn.Module):
    """Soft per-coordinate std floor, optionally plus mean squared correlation.

    No mean anchor, upper variance bound, or prescribed distribution shape.
    Compute statistics across batch separately at each time, as in SIGReg.
    Correlation (not covariance) discourages duplicate directions without
    directly rewarding smaller scales. Its relative coefficient is fixed at 1.
    """
    def __init__(self, decorrelate=False):
        super().__init__()
        self.decorrelate = decorrelate

    def forward(self, z):
        if z.ndim != 3 or z.shape[1] < 2 or z.shape[2] < 2:
            raise ValueError("Expected (time, batch>=2, latent>=2)")
        z = z.float()
        centered = z - z.mean(1, keepdim=True)
        variance = centered.square().sum(1) / (z.shape[1]-1)
        floor = torch.relu(1 - (variance + 1e-4).sqrt()).square().mean()
        if not self.decorrelate:
            return floor
        standardized = centered / variance.clamp_min(1e-4).sqrt().unsqueeze(1)
        corr = standardized.transpose(-1, -2) @ standardized / (z.shape[1]-1)
        offdiag = corr - torch.diag_embed(corr.diagonal(dim1=-2, dim2=-1))
        d = z.shape[-1]
        return floor + offdiag.square().sum((-1, -2)).mean() / (d*(d-1))


class NoRegularizer(nn.Module):
    def forward(self, z):
        return z.new_zeros(())


class RadialRegularizer(DistributionRegularizer):
    """Uniform direction times R, with R^2 ~ Gamma(D/(2v), scale=2v).

    E[R^2]=D and Cov(X)=I. v=1 is exactly Gaussian; v=0 is the
    radius-sqrt(D) sphere surface. v scales Var(R^2) relative to Gaussian.
    Targets are precomputed once in float64 with SciPy, then stored in fp32.
    Only the projected CF target changes; no explicit radial loss is added.
    """
    def __init__(self, relative_variance=1., latent_dim=192, **kwargs):
        if not math.isfinite(relative_variance) or relative_variance < 0:
            raise ValueError("relative_variance must be finite and nonnegative")
        if not isinstance(latent_dim, int) or latent_dim < 2:
            raise ValueError("latent_dim must be an integer >= 2")
        super().__init__(kind="gaussian", **kwargs)
        self.relative_variance = relative_variance
        self.latent_dim = latent_dim
        self.kind = "radial"
        if relative_variance == 1:
            target = self.phi.clone()  # Exact SIGReg fp32 reference.
        else:
            from scipy.special import hyp0f1, hyp1f1
            t = self.t.double().numpy()
            if relative_variance == 0:
                values = hyp0f1(latent_dim/2, -latent_dim*t*t/4)
            else:
                k = latent_dim/(2*relative_variance)
                values = hyp1f1(k, latent_dim/2, -latent_dim*t*t/(4*k))
            target = torch.as_tensor(values, dtype=self.t.dtype)
        if not torch.isfinite(target).all() or (target.abs() > 1.000001).any():
            raise RuntimeError("Invalid radial characteristic function")
        self.register_buffer("radial_phi", target)

    def target_cf(self, directions):
        if directions.shape[0] != self.latent_dim:
            raise ValueError("Radial target dimension does not match embedding")
        return self.radial_phi.expand(directions.shape[1], -1)


RADIAL_VARIANCES = {"radial_broad": 8., "radial_half": .5,
                    "radial_quarter": .25, "radial_sphere": 0.}


def make_regularizer(kind, num_proj=64, separation=.75, knots=17, latent_dim=192):
    if kind in {"variance_floor", "variance_decorrelation"}:
        return VarianceFloorRegularizer(decorrelate=kind == "variance_decorrelation")
    if kind in RADIAL_VARIANCES:
        return RadialRegularizer(relative_variance=RADIAL_VARIANCES[kind],
                                 latent_dim=latent_dim, num_proj=num_proj, knots=knots)
    if kind in {"student_t3", "student_t5", "student_t9"}:
        return StudentTRegularizer(df=int(kind.removeprefix("student_t")), num_proj=num_proj, knots=knots)
    if kind == "none":
        return NoRegularizer()
    if kind == "moments":
        return MomentRegularizer()
    return DistributionRegularizer(kind, separation=separation, num_proj=num_proj, knots=knots)
