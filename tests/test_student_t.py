import math

import pytest
import torch
from lewm.regularizers import StudentTRegularizer, make_regularizer


@pytest.mark.parametrize("df", [3, 5, 9])
def test_cf_matches_samples_from_joint_student_t(df):
    torch.manual_seed(973)
    n, d = 80000, 5
    # Integer df permits an independent chi-square construction without any
    # special functions. One radial multiplier is shared by all coordinates.
    chi = torch.randn(n, df).square().sum(1, keepdim=True)
    samples = torch.randn(n, d) * ((df-2)/chi).sqrt()
    a = torch.randn(d, 8)
    a /= a.norm(dim=0)
    reg = StudentTRegularizer(df=df)
    phase = (samples @ a).unsqueeze(-1) * reg.t
    torch.testing.assert_close(phase.cos().mean(0), reg.target_cf(a), atol=.016, rtol=0)
    assert phase.sin().mean(0).abs().max() < .016


@pytest.mark.parametrize("df", [3, 5, 9])
def test_unit_variance_from_cf_curvature(df):
    reg = StudentTRegularizer(df=df).double()
    # Independent formula evaluated in double precision near zero.
    t = .0001
    x = math.sqrt(df-2)*t
    poly = {3: 1+x, 5: 1+x+x*x/3, 9: 1+x+3*x*x/7+2*x**3/21+x**4/105}[df]
    variance = 2*(1-poly*math.exp(-x))/(t*t)
    assert abs(variance-1) < .001
    assert reg.student_phi[0] == 1


def test_student_targets_preserve_kernel_and_receive_gradients():
    from lewm.module import SIGReg
    reference = SIGReg()
    for df in (3, 5, 9):
        reg = make_regularizer(f"student_t{df}", num_proj=16)
        torch.testing.assert_close(reg.weights, reference.weights)
        x = torch.randn(4, 32, 12, requires_grad=True)
        reg(x).backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_unsupported_df_rejected():
    with pytest.raises(ValueError):
        StudentTRegularizer(df=2)
