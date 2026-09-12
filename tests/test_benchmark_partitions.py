import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("partitions", Path(__file__).parents[1]/"experiments/benchmark/partitions.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partitions_are_disjoint_repeatable_and_goals_stay_inside_episodes():
    lengths = np.arange(20, 120)
    a = module.make_partition(lengths)
    assert a == module.make_partition(lengths)
    sets = [set(a[key]) for key in ("train", "validation", "test")]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert set.union(*sets) == set(np.flatnonzero(lengths >= 30))
    for key in ("validation", "test"):
        for ep, start in zip(a[key+"_goals"]["episodes"], a[key+"_goals"]["starts"]):
            assert ep in a[key]
            assert 0 <= start < start+25 < lengths[ep]


def test_missing_seed_adapter_is_repeatable_and_preserves_recorded_seeds():
    class Dummy:
        column_names = ["pixels"]
        def load_chunk(self, episodes_idx, start, end):
            return [{"pixels": torch.zeros(3, 3, 4, 4)} for ep in episodes_idx]
    wrapped = module.EvaluationSeeds(Dummy())
    assert wrapped.column_names == ["pixels", "seed"]
    chunks = wrapped.load_chunk([2, 7], [0, 0], [3, 3])
    assert chunks[0]["seed"].tolist() == [6175]*3
    assert chunks[1]["seed"].tolist() == [6180]*3
    dummy = Dummy()
    dummy.column_names = ["pixels", "seed"]
    assert not module.EvaluationSeeds(dummy).injected
