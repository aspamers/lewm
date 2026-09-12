# Training-only encoder projection BatchNorm calibration

New planning evaluations calibrate the encoder projection's single BatchNorm
before using a checkpoint. The fixed recipe draws 512 clips (seed 8821) from
the existing training partition, with batch 64 and four frames per clip.
Exactly the same calibration clips are used for all methods in an environment.

Float64 streaming moment accumulation estimates the input mean and unbiased
variance, then stores these as float32 BatchNorm buffers. Encoding is float32,
TF32 is disabled, and all layers are in evaluation mode during calibration.
No gradients, weights, predictor buffers, validation images or test images are
used to adapt the model. Calibrated buffers are in memory only; original weight
files are preserved.

Evaluation outputs use distinct `test-calibrated.json` and
`validation-calibrated.json` filenames. Each contains checkpoint, training
partition, calibration-code and evaluation-code hashes, sampling settings,
planning settings/goals, and a hash of the calibrated statistics. Cache reuse
requires matching provenance. The original uncalibrated files remain intact.
Embedding diagnostics follow the calibration metadata of the planning results
and check that their calibrated statistics match.

`reevaluate_calibrated.py` reevaluates the original 500-update selected models in
both environments, the completed 1,000-update TwoRoom models, and the three
completed seed-31 full-precision floor-weight trials. It does not resume training
or reselect coefficients. The cohorts differ in training budget, precision and
loss; compare methods only within matched cohorts. The floor-weight trials have
only one seed. Repeated use of these goals makes this exploratory follow-up,
even though the calibration procedure itself uses training data exclusively.

Tests verify streaming moment accuracy, unchanged model parameters and predictor
statistics, restored module modes, independence from evaluation batch grouping,
and cleanup/state preservation on failed calibration.
