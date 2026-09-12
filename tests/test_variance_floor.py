import pytest
import torch

from lewm.regularizers import VarianceFloorRegularizer


def orthogonal_samples():
    # Walsh columns: exactly centered, orthogonal, sample std above one.
    return torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])[None]


def test_floor_accepts_non_gaussian_and_redundant_embeddings():
    z = orthogonal_samples()
    floor = VarianceFloorRegularizer()
    assert floor(z) == 0
    assert floor(z*5+17) == 0  # No ceiling or mean anchor.
    duplicate = z[:, :, :1].expand(-1, -1, 2)
    assert floor(duplicate) == 0  # Variance alone cannot reject rank-one codes.


def test_decorrelation_distinguishes_orthogonal_and_duplicate_codes():
    reg = VarianceFloorRegularizer(decorrelate=True)
    z = orthogonal_samples()
    assert reg(z) < 1e-12
    duplicate = z[:, :, :1].expand(-1, -1, 2)
    torch.testing.assert_close(reg(duplicate), torch.tensor(1.))
    torch.testing.assert_close(reg(duplicate*5+17), reg(duplicate))


@pytest.mark.parametrize("decorrelate", [False, True])
def test_floor_penalizes_collapse_and_encourages_small_variance(decorrelate):
    reg = VarianceFloorRegularizer(decorrelate=decorrelate)
    assert reg(torch.ones(2, 8, 4)) > .9
    z = (.1*orthogonal_samples()).requires_grad_()
    loss = reg(z)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert (z.grad*z).sum() < 0  # Descent increases spread.
    # Different constants at each time must not masquerade as batch variance.
    per_time_constants = torch.arange(4.)[:, None, None].expand(4, 8, 2)
    assert reg(per_time_constants) > .9


@pytest.mark.parametrize("decorrelate", [False, True])
def test_exact_collapse_has_finite_gradient_and_bad_shape_rejected(decorrelate):
    reg = VarianceFloorRegularizer(decorrelate=decorrelate)
    z = torch.zeros(2, 8, 4, requires_grad=True)
    reg(z).backward()
    assert torch.isfinite(z.grad).all()  # Exact symmetry is stationary, not rescued.
    with pytest.raises(ValueError):
        reg(torch.zeros(2, 1, 4))
