"""Predeclared grids and exposure-aware goal reservation, without image access."""
import numpy as np

TUNING_SEEDS = [1101,1102]
CONFIRMATION_SEEDS = [2101,2102,2103]
CONFIGS = [{"name":f"gaussian-w{w:g}","kind":"gaussian","weight":w}
           for w in (.01,.03,.09,.27)] + [
    {"name":f"floor{f:g}-corr{c:g}","kind":"variance_decorrelation","weight":1.,
     "floor_weight":f,"correlation_weight":c} for f in (.3,1.) for c in (.03,.1)]


def fresh_partition(old, lengths):
    lengths = np.asarray(lengths)
    # Replay the episode draws from the only historical test-embedding probe:
    # diagnostics.py, Clips(old test, seed=8802, steps=8, batch=64). Starts do not
    # affect episode exposure. Exclude entire episodes, not only sampled frames.
    episodes = np.asarray(old["test"])
    counts = lengths[episodes]-19
    diagnostic = np.random.default_rng(8802).choice(episodes,size=512,p=counts/counts.sum())
    exposed = set(map(int,diagnostic)) | set(old["test_goals"]["episodes"]) | {0}
    untouched = sorted(set(old["test"])-exposed)
    if len(untouched) < 64:
        raise ValueError("Not enough previously uninspected test episodes for 64 distinct goals")
    rng = np.random.default_rng(66391)
    def goals(pool, n):
        chosen = rng.choice(pool,size=n,replace=False)
        return {"episodes":chosen.tolist(),"starts":[int(rng.integers(lengths[ep]-25)) for ep in chosen]}
    result = {**old,"test":untouched,"validation_goals":goals(old["validation"],32),
              "test_goals":goals(untouched,64),
              "fresh_protocol":{"goal_seed":66391,"excluded_prior_test_episodes":sorted(exposed),
                                "prior_test_goals_are_development_only":True,
                                "prior_diagnostic_seed":8802,"prior_diagnostic_clips":512}}
    return result
