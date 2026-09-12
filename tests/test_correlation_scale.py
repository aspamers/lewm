import torch
from lewm.regularizers import VarianceFloorRegularizer


def correlation(z):
    return VarianceFloorRegularizer(True)(z)-VarianceFloorRegularizer(False)(z)


def test_shrinking_correlated_features_cannot_erase_penalty():
    x = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1)
    x = (x-x.mean()).expand(2, 32, 8)
    for scale in (1., .001, .00001):
        torch.testing.assert_close(correlation(x*scale), torch.tensor(1.), atol=2e-6, rtol=0)


def test_low_scale_penalty_has_no_radial_shrink_incentive():
    torch.manual_seed(45)
    x = (torch.randn(2, 32, 8)*1e-4).requires_grad_()
    penalty = correlation(x)
    gradient, = torch.autograd.grad(penalty, x)
    assert torch.isfinite(gradient).all()
    assert abs(float((gradient*x).sum())) < 1e-5


def test_constant_and_mixed_constant_features_have_finite_gradients():
    for constant in (True, False):
        x = torch.zeros(2, 32, 8)
        if not constant:
            x[..., 0] = torch.arange(32)*1e-5
        x.requires_grad_()
        loss = VarianceFloorRegularizer(True)(x)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_cpu_autocast_does_not_change_moment_penalty():
    x = torch.randn(2, 32, 8)
    regularizer = VarianceFloorRegularizer(True)
    expected = regularizer(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = regularizer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
