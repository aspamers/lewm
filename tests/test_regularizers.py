import math

import pytest
import torch

from lewm.module import SIGReg
from lewm.regularizers import DistributionRegularizer, MomentRegularizer


def test_gaussian_is_exact_upstream_sigreg():
    x = torch.randn(4, 32, 24)
    torch.manual_seed(17)
    expected = SIGReg(num_proj=32)(x)
    torch.manual_seed(17)
    actual = DistributionRegularizer(num_proj=32)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mixture_cf_matches_samples_from_one_joint_distribution():
    torch.manual_seed(3)
    d, n = 8, 60000
    r = .75
    sign = (torch.randint(2, (n, 1)) * 2 - 1).float()
    samples = sign * math.sqrt(r) + torch.randn(n, d) * math.sqrt(1-r)
    directions = torch.randn(d, 12)
    directions /= directions.norm(dim=0)
    reg = DistributionRegularizer(kind="mixture")
    phase = (samples @ directions).unsqueeze(-1) * reg.t
    torch.testing.assert_close(phase.cos().mean(0), reg.target_cf(directions), atol=.015, rtol=0)
    assert phase.sin().mean(0).abs().max() < .015


def test_covariance_match_does_not_erase_distribution_difference():
    directions = torch.ones(8, 1) / math.sqrt(8)
    mixture = DistributionRegularizer(kind="mixture")
    matched = DistributionRegularizer(kind="matched_gaussian")
    assert (mixture.target_cf(directions) - matched.target_cf(directions)).abs().max() > .1
    # Both have zero mean and variance (1-r) + r*(sum a_i)^2 along a.
    for kind in ("mixture", "matched_gaussian"):
        reg = DistributionRegularizer(kind=kind, separation=0)
        torch.testing.assert_close(reg.target_cf(directions), reg.phi[None])


@pytest.mark.parametrize("kind", ["gaussian", "mixture", "matched_gaussian", "moments"])
def test_regularizer_pushes_near_collapse_and_backpropagates(kind):
    x = (torch.randn(4, 32, 16) * .02).requires_grad_()
    reg = MomentRegularizer() if kind == "moments" else DistributionRegularizer(kind=kind, num_proj=16)
    loss = reg(x)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_moments_allow_non_gaussian_shapes():
    # Eight balanced discrete points have identity sample covariance after scaling.
    x = torch.tensor([[a, b, c] for a in (-1., 1.) for b in (-1., 1.) for c in (-1., 1.)])
    x *= math.sqrt(7/8)
    assert MomentRegularizer()(x[None]) < 1e-8
