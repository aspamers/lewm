from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lewm.bottleneck import (BottleneckPredictor, ObservationDecoder,
                              PredictiveBottleneck, predictive_loss)
from lewm.module import Embedder, MLP, SIGReg

torch.set_num_threads(2)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 8 * 8, 24)

    def forward(self, x, **kwargs):
        return SimpleNamespace(last_hidden_state=self.linear(x.flatten(1))[:, None])


def model(width=16):
    return PredictiveBottleneck(
        encoder=Encoder(), projector=MLP(24, 32, width),
        predictor=BottleneckPredictor(width, hidden_dim=24, num_frames=3,
                                      depth=1, heads=2, dim_head=12, mlp_dim=32),
        pred_proj=MLP(24, 32, width), action_encoder=Embedder(2, emb_dim=24),
        decoder=ObservationDecoder(width, image_size=8, hidden_dim=8))


def batch():
    return {"pixels": torch.randn(4, 8, 3, 8, 8), "action": torch.randn(4, 8, 2)}


@pytest.mark.parametrize("width", [16, 32])
def test_observation_gradient_reaches_all_components(width):
    m = model(width)
    # AdaLN-Zero intentionally blocks conditioning gradients on the first
    # update. Let its zero-initialized modulation learn before checking actions.
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
    predictive_loss(m, batch(), SIGReg(), latent_weight=0)["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()
    losses = predictive_loss(m, batch(), SIGReg(), latent_weight=0)
    losses["loss"].backward()
    for component in [m.encoder, m.projector, m.predictor, m.action_encoder, m.pred_proj, m.decoder]:
        assert sum(p.grad.abs().sum() for p in component.parameters() if p.grad is not None) > 0


class ExactDynamics:
    decoder = nn.Identity()

    def encode(self, b):
        return {"emb": b["pixels"].flatten(2), "act_emb": b["action"]}

    def predict(self, z, a):
        return z + a


def test_open_loop_alignment_and_no_future_teacher_forcing():
    class Decode(nn.Module):
        def forward(self, z):
            return z[:, :, None, None].expand(-1, -1, 4, 4)

    class Dynamics(ExactDynamics):
        decoder = Decode()
        def encode(self, b):
            return {"emb": b["pixels"][:, :, :, 0, 0], "act_emb": b["action"]}

    actions = torch.arange(1., 9.).reshape(1, 8, 1)
    positions = torch.cat([torch.zeros(1, 1, 1), actions[:, :-1].cumsum(1)], dim=1)
    b = {"pixels": positions[..., None, None].expand(1, 8, 1, 4, 4).clone(), "action": actions}
    assert predictive_loss(Dynamics(), b, None)["obs_loss"] == 0
    # Alter intermediate future frames which are NOT scored at horizons 1,3,5.
    # A teacher-forced rollout would now make errors at later scored horizons.
    b["pixels"][:, [4, 6]] = -999
    assert predictive_loss(Dynamics(), b, None)["obs_loss"] == 0


def test_baseline_matches_original_one_step_objective():
    m, b = model(), batch()
    out = m.encode(dict(b))
    expected = (m.predict(out["emb"][:, :3], out["act_emb"][:, :3]) - out["emb"][:, 1:4]).square().mean()
    actual = predictive_loss(m, b, None, observation_weight=0, latent_weight=1)
    torch.testing.assert_close(actual["loss"], expected)


def test_invalid_windows_and_disabled_regularizer():
    m, b = model(), batch()
    predictive_loss(m, b, None)  # disabled means never called
    b["action"][0, 2, 0] = float("nan")
    with pytest.raises(ValueError, match="episode boundaries"):
        predictive_loss(m, b, None)
    with pytest.raises(ValueError, match="Need"):
        predictive_loss(m, {"pixels": b["pixels"][:, :4], "action": b["action"]}, None)


def test_decoded_cost_uses_real_goal():
    m = model()
    pred = torch.randn(2, 3, 5, 16)
    goal = m.decoder(pred[:, :, -1].reshape(6, 16)).reshape(2, 3, 1, 3, 8, 8)
    torch.testing.assert_close(m.decoded_terminal_cost(pred, goal), torch.zeros(2, 3))
    assert (m.decoded_terminal_cost(pred, goal + 1) > 0).all()


def test_episode_split_and_training_only_statistics():
    from lewm.experiment_data import split_episodes, action_statistics
    class Dataset:
        clip_indices = [(ep, t) for ep in range(4) for t in range(3)]
        offsets = [0, 3, 6, 9]
        lengths = [3] * 4
        def get_col_data(self, name):
            return torch.arange(12.).reshape(-1, 1)
    ds = Dataset()
    train, val, eps = split_episodes(ds, .5, 42)
    assert set(ds.clip_indices[i][0] for i in train.indices).isdisjoint(
        ds.clip_indices[i][0] for i in val.indices)
    mean, std = action_statistics(ds, eps)
    values = torch.cat([torch.arange(ep*3., ep*3.+3) for ep in eps])
    torch.testing.assert_close(mean.flatten(), values.mean()[None])
    torch.testing.assert_close(std.flatten(), values.std()[None])
