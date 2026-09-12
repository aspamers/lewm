import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch

sys.path.insert(0, str(Path("experiments").resolve()))
from regularizer_study import WIDTH, dataset, model, planning, training_loss, transition


def test_training_has_no_decoder_and_ignores_future_beyond_next_target():
    m = model().eval()
    assert not hasattr(m, "decoder")
    b, _ = dataset(1, 4)
    expected = training_loss(m, b, None, 0)
    b["pixels"][:, 4:] = 999
    b["action"][:, 4:] = 999
    torch.testing.assert_close(training_loss(m, b, None, 0), expected)


def test_cem_with_oracle_dynamics_improves_on_staying_and_random():
    # Replace pixels by the known state only for this planner unit test.
    # The actual experiment never gives simulator states to the model/planner.
    class Oracle:
        def encode(self, b):
            return {"emb": b["oracle_z"]}
        def action_encoder(self, a):
            return a
        def predict(self, z, a):
            result = z.clone()
            result[..., :2] = transition(z[..., :2], a)
            return result
    data, states = dataset(24000, 128)
    data["oracle_z"] = torch.zeros(128, 8, WIDTH)
    data["oracle_z"][..., :2] = states
    result = planning(Oracle(), data, states)
    assert result["goal_distance"] < result["stay_distance"] * .5
    assert result["goal_distance"] < result["random_distance"] * .5


def test_all_full_model_configs_resolve_regularizers():
    with initialize_config_dir(config_dir=str(Path("config/train").resolve()), version_base=None):
        for kind in ("gaussian", "mixture", "matched_gaussian", "moments", "none"):
            cfg = compose(config_name="regularizer_study", overrides=[f"target={kind}"])
            assert cfg.model._target_ == "lewm.jepa.JEPA"
            assert cfg.embed_dim == 192 and cfg.num_preds == 1
            reg = instantiate(cfg.regularizer)
            assert torch.isfinite(reg(torch.randn(4, 8, 192)))
