from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch
from torch import nn


def test_benchmark_configs_preserve_hidden_width():
    with initialize_config_dir(config_dir=str(Path("config/train").resolve()), version_base=None):
        for width in (16, 32):
            cfg = compose(config_name="bottleneck", overrides=[f"latent_dim={width}", "data=tworoom"])
            cfg.model.action_encoder.input_dim = 10
            cfg.img_size = 28
            model = instantiate(cfg.model, encoder=nn.Identity())
            assert model.projector.net[0].in_features == 192
            assert model.projector.net[-1].out_features == width
            assert model.predictor.core.transformer.norm.normalized_shape == (192,)
            action = model.action_encoder(torch.randn(2, 3, 10))
            assert action.shape == (2, 3, 192)
            prediction = model.predict(torch.randn(2, 3, width), action)
            assert prediction.shape == (2, 3, width)
            assert model.decoder(prediction[:, -1]).shape == (2, 3, 28, 28)
