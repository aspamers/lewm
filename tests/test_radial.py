import math

import pytest
import torch

from lewm.regularizers import RadialRegularizer, DistributionRegularizer


@pytest.mark.parametrize("v", [0., .25, .5, 1., 8.])
def test_radial_cf_against_independent_joint_samples(v):
    # Gamma draw independent of uniform direction. Test actual D=192.
    torch.manual_seed(7291)
    d, n = 192, 50000
    direction = torch.randn(n, d)
    direction /= direction.norm(dim=1, keepdim=True)
    r2 = torch.full((n,), float(d)) if v == 0 else torch.distributions.Gamma(d/(2*v), 1/(2*v)).sample((n,))
    x = direction[:, :3] * r2.sqrt()[:, None]
    reg = RadialRegularizer(relative_variance=v, latent_dim=d)
    phase = x[..., None] * reg.t
    torch.testing.assert_close(phase.cos().mean(0), reg.radial_phi.expand(3, -1), atol=.015, rtol=0)
    assert phase.sin().mean(0).abs().max() < .015
    assert abs(float(x.square().mean()) - 1) < .02
    assert abs(float(r2.mean())/d - 1) < .01
    if v:
        assert abs(float(r2.var())/(2*d*v) - 1) < .035
    else:
        assert r2.var() == 0


def test_gaussian_reference_identical_loss_and_gradient():
    z = torch.randn(4, 32, 192, requires_grad=True)
    reference = DistributionRegularizer(num_proj=16)
    radial = RadialRegularizer(relative_variance=1, num_proj=16)
    torch.manual_seed(53)
    a = reference(z)
    ga = torch.autograd.grad(a, z)[0]
    torch.manual_seed(53)
    b = radial(z)
    gb = torch.autograd.grad(b, z)[0]
    assert torch.equal(a, b) and torch.equal(ga, gb)


def test_radial_cf_closed_forms_in_two_dimensions():
    # D=2, k=1/2 => 1F1(1/2;1;-t^2)=exp(-t^2/2)*I0(t^2/2).
    broad = RadialRegularizer(relative_variance=2, latent_dim=2)
    expected = torch.special.i0e(broad.t.double().square()/2)
    torch.testing.assert_close(broad.radial_phi.double(), expected, atol=5e-8, rtol=1e-7)
    # Circle projection CF J0(sqrt(2)t), independent Bessel reference.
    sphere = RadialRegularizer(relative_variance=0, latent_dim=2)
    expected = torch.special.bessel_j0(math.sqrt(2)*sphere.t.double())
    torch.testing.assert_close(sphere.radial_phi.double(), expected, atol=5e-8, rtol=1e-7)


def test_radial_targets_receive_gradients_and_reject_wrong_dimension():
    for v in (0., .25, .5, 8.):
        z = torch.randn(4, 32, 192, requires_grad=True)
        reg = RadialRegularizer(relative_variance=v, num_proj=8)
        reg(z).backward()
        assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
        with pytest.raises(ValueError):
            reg(torch.randn(4, 32, 12))
    for v in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RadialRegularizer(relative_variance=v)
