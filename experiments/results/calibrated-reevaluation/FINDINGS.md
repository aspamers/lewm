# Interpretation of the calibrated planning reevaluation

Fifteen saved checkpoints were reevaluated. Training-only encoder BatchNorm
calibration is now integrated into evaluation; no weights were retrained.
Original scores are preserved. See REPORT.md for every individual seed.

For the original 500-update checkpoints, calibrated TwoRoom means are 62.5%
for Gaussian (56.25%, 68.75%) and 21.875% for the old combined regularizer
(12.5%, 31.25%). The random baseline is 37.5%. The earlier uncalibrated lead
for the combined method does not survive correction of this evaluation issue.

At 1,000 updates on TwoRoom, Gaussian obtains 100% and 62.5% (mean 81.25%),
while the old combined method obtains 31.25% for each seed. These original
models still reflect the earlier mixed-precision training and loss formulation.
Calibration does not undo a representation learned with those shortcomings.

On PushT at 500 updates, Gaussian remains at zero for both seeds. The old
combined method achieves 6.25% for each seed after calibration. That is one
successful goal per model on a shared 16-goal set, not broad evidence of
successful PushT control or independent replicated task generalization.

The three corrected, full-precision 250-update floor-strength models, all seed
31, obtain 31.25%, 56.25%, and 43.75% at floor weights 0.1, 0.3, and 1, with
correlation coefficient fixed at 0.1. This is preliminary evidence that floor
strength matters and that making spread closest to one need not maximize
planning performance. It does not identify a robust optimum. There is no
matched 250-update full-precision Gaussian control or second stronger-floor
seed yet; do not compare these numbers as a controlled superiority claim.

Both interrupted training sweeps remain stopped. Next comparisons need matched
precision, budget, calibration and seeds, with fresh confirmation tasks after
these exploratory, repeatedly inspected goals.
