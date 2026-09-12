# Independent floor-strength diagnostic

Question: after correcting precision, does the weak variance-floor weight explain
the remaining low embedding spread, or does a stronger floor merely harm prediction?

* TwoRoom only, full repository architecture, float32 training and inference;
  TF32 disabled, encoder activation checkpointing enabled.
* Objective: prediction MSE + floor_weight * floor + 0.1 * correlation penalty.
* Floor weights: 0.1, 0.3, 1, 3. Both seed 31 and seed 32 run each weight.
* 250 updates, batch 64, AdamW 5e-5, decay 1e-3, gradient clip 1.
* Match initialization and minibatch sequence across weights within each seed.
* The 0.1/0.1 arm repeats the previous corrected full-precision condition.
* Four fixed validation batches (seed 7791, 256 clips) are probed every 50 updates.
  These are repeated descriptive measurements, not independent task trials.
* Measure batch-statistics and evaluation spread, each penalty, prediction MSE
  divided by target variance, similarly normalized persistence error, and error
  after deterministic action shuffling. Restore BatchNorm buffers after probing.
* No planning test goals are used, no new best coefficient is declared from test
  performance, and no training budget is adapted to intermediate outcomes.

The runner writes progress, per-run curves and results to outputs/floor-strength.
After all eight runs finish, it generates REPORT.md and exports the compact
bundle to experiments/results/floor-strength. Partial runs are restarted from
initialization on rerun; completed arms are reused only if the manifest matches.

The two-update integration smoke check passed, including the predictor-based
diagnostics. Unit tests verify that changing the floor weight preserves the
independent correlation contribution.
