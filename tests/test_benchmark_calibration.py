import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location("calibration",Path(__file__).parents[1]/"experiments/benchmark/calibration.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.BatchNorm1d(5),torch.nn.Linear(5,2))
        self.predictor = torch.nn.BatchNorm1d(2)
    def encode(self,batch):
        return self.projector(batch["pixels"])


def test_calibration_matches_training_population_and_only_changes_target_statistics():
    torch.manual_seed(55)
    model = Toy().train()
    model.predictor.eval()
    modes = {name:m.training for name,m in model.named_modules()}
    before = {k:v.clone() for k,v in model.state_dict().items()}
    x = torch.randn(19,3)+2
    expected = model.projector[0](x).detach()
    metadata = module.calibrate_projection_bn(model,[{"pixels":x[:4]},{"pixels":x[4:]}])
    torch.testing.assert_close(model.projector[1].running_mean,expected.mean(0))
    torch.testing.assert_close(model.projector[1].running_var,expected.var(0))
    assert metadata["observations"] == 19
    for key,value in model.state_dict().items():
        if key not in ("projector.1.running_mean","projector.1.running_var"):
            torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    assert modes == {name:m.training for name,m in model.named_modules()}
    model.eval()
    torch.testing.assert_close(model.encode({"pixels":x}),torch.cat([model.encode({"pixels":x[:4]}),model.encode({"pixels":x[4:]})]))


def test_failed_calibration_preserves_state_and_removes_hook():
    model = Toy().train()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError):
        module.calibrate_projection_bn(model,[{"pixels":torch.full((4,3),float("nan"))}])
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    assert model.training
    assert not model.projector[1]._forward_pre_hooks
