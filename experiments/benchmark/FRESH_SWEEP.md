# Fresh matched retuning and confirmation: 200-update pilot

The user requested a quicker 200-update first pass. Both methods use this same
budget for tuning and confirmation; no settings receive extra training based on
outcomes. TwoRoom and PushT remain the only target environments.

## Predeclared search

* Gaussian weights: 0.01, 0.03, 0.09, 0.27.
* Corrected floor/correlation: floor 0.3 or 1 crossed with correlation 0.03 or 0.1.
* Four settings per method, paired tuning seeds 1101 and 1102, 32 validation goals
  per environment. This is a coarse bounded search, not exhaustive optimization.
* Select by mean validation success; ties follow the listed configuration order.
* Freeze both environments' choices before any confirmation evaluation.
* Train selected settings from scratch with fresh seeds 2101, 2102, 2103.
* Evaluate on 64 distinct, previously uninspected goal episodes per environment.
  All 12 confirmation models finish training before any final outcomes are opened.

This totals 32 tuning runs plus 12 confirmation runs, each 200 updates at batch
64. All use the full architecture, AdamW lr 5e-5, weight decay 1e-3, clipping 1,
float32, TF32 disabled, encoder activation checkpointing, and encoder projection
BatchNorm calibration from the same 512 training clips. CEM remains 64 candidates,
three iterations, horizon 5, action block 5 and a 50-step episode budget.

## Data and exposure

Training episodes and train-only action normalization remain unchanged. The
previous validation set stays development data. Previously inspected test goals
are development-only and never appear in confirmation. We also replay and exclude
the entire episodes selected by the historical test embedding diagnostic (seed
8802, 512 clips), plus episode zero used for simulator reconstruction checks.

`fresh_protocol.py` reserves fresh goals from the remaining original test pool
using seed 66391. It reads episode lengths, not images or model outcomes. Partitions
record excluded episodes, selected goals and the recipe. This is fresh confirmation
within the same source datasets, not a new environment distribution. Model seeds
are never chosen by performance. Confirmation goals are shared across model seeds,
so individual goal outcomes must not be treated as independent across seeds.

The `--smoke` path trains two-update models and evaluates validation goals only.
It never opens confirmation goals. Full run command:

```text
python experiments/benchmark/fresh_sweep.py --steps 200 --output outputs/fresh-sweep-200
```

Source/settings are frozen by manifest; altered recipes require a fresh output
directory. Completed runs and optimizer checkpoints support resuming interruptions.
The report and raw results are exported automatically to experiments/results/fresh-sweep-200.
