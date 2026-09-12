import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("fresh_protocol",Path(__file__).parents[1]/"experiments/benchmark/fresh_protocol.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_new_goals_exclude_all_previously_probed_test_episodes():
    lengths = np.arange(2000)%80+40
    old = {"train":list(range(500)),"validation":list(range(500,700)),"test":list(range(700,2000)),
           "test_goals":{"episodes":list(range(700,716)),"starts":[0]*16}}
    fresh = module.fresh_partition(old,lengths)
    assert fresh == module.fresh_partition(old,lengths)
    exposed = set(fresh["fresh_protocol"]["excluded_prior_test_episodes"])
    assert not exposed & set(fresh["test"])
    assert fresh["train"] == old["train"]
    assert not set(fresh["test"]) & set(fresh["train"]+fresh["validation"])
    assert len(set(fresh["test_goals"]["episodes"])) == 64
    for ep,start in zip(fresh["test_goals"]["episodes"],fresh["test_goals"]["starts"]):
        assert ep in old["test"] and ep not in exposed
        assert 0 <= start < start+25 < lengths[ep]


def test_equal_search_counts_and_disjoint_seed_sets():
    assert sum(c["kind"]=="gaussian" for c in module.CONFIGS) == 4
    assert sum(c["kind"]=="variance_decorrelation" for c in module.CONFIGS) == 4
    assert not set(module.TUNING_SEEDS) & set(module.CONFIRMATION_SEEDS)
