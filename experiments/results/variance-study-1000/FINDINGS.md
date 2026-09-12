# A variance floor prevents collapse; decorrelation improves planning

Completed 50 synthetic runs: five paired seeds, 1,000 updates, three coefficient
settings per active regularizer, plus no-regularizer control. Architecture and
prediction loss stayed fixed. Coefficients were chosen on validation before
evaluating fresh test seed 72000. No Gaussian or mixture shape is prescribed
by either new regularizer.

| Arm | Selected weight | Test goal distance (lower better) | Success | Mean coordinate std | Correlation effective rank |
|---|---|---|---|---|---|
| None | 0 | 0.40308 | 5.8% | 0.0030 | 35.60 |
| Variance floor | 3 | 0.13214 | 50.8% | 1.1604 | 2.15 |
| Floor + decorrelation | 0.3 | 0.10810 | 65.0% | 1.0270 | 9.73 |
| SIGReg | 0.09 | 0.10273 | 65.0% | 1.0397 | 14.28 |

## What worked

The variance floor successfully prevented near-constant collapse: all dimensions
had test standard deviation at least 0.9. It improved planning substantially
over no regularizer. However, mean absolute coordinate correlation was 0.678,
and the largest correlation eigenvalue held 66.8% of the trace. Active
coordinates largely repeated a small number of signals.

Floor-only mean goal distance was 0.02942 worse than SIGReg; paired five-seed
bootstrap 95% interval [+0.00519, +0.05364]. This supports a planning disadvantage
for the floor alone under this recipe, not a claim that it failed to prevent
collapse or could never work with another model.

Adding decorrelation raised correlation effective rank from 2.15 to 9.73 and
reduced mean absolute correlation to 0.267. Its goal distance improved by
0.02405 relative to floor-only (exploratory interval [-0.05215, -0.00054]).
It matched SIGReg's mean threshold success of 65%, with mean distance 0.00537
worse and interval [-0.00692, +0.01867]. That is a promising simpler candidate,
not established equivalence or superiority. Both are soft constraints.

No regularizer had a seemingly high relative rank but total variance only
0.00179; all dimensions had std below 0.1. Effective rank without absolute
variance is therefore misleading as a collapse diagnostic.

## Useful state information versus planning geometry

The floor variants did not lose physical position information. Position probe
MSE was 0.000851 for floor, 0.000778 for combined, and 0.001177 for SIGReg.
Five-step position probe MSE was 0.01295, 0.01088, and 0.02494 respectively.
Action shuffling degraded predictions in all three active arms.

Thus the floor's planning deficit is not explained simply by absence of a
decodable position signal. A diagnostic linear probe can compensate for
embedding geometry, while the planner uses raw Euclidean latent distance.
Redundancy and geometry are plausible contributors, but this experiment does
not isolate their causal role from changes in optimization and dynamics.

## Limits and next question

The combined arm selected the lowest coefficient tested, 0.3. Its useful range
may extend lower. The internal decorrelation-to-floor coefficient was fixed at
1, not separately tuned. Five seeds and 24 reused test goals are a small pilot;
bootstrap intervals exclude dataset and coefficient-selection uncertainty.
Secondary intervals have no multiple-comparison correction. This is not the
full LeWM/TwoRoom benchmark. The new test split prevents direct numerical
comparison with previous tables.

The evidence supports testing the shape-free floor+decorrelation candidate
further, with a refined validation grid and independent replication. It does
not support the stronger claim that per-coordinate variance alone is enough
to match SIGReg's planning performance. No follow-up has been run yet.

All 45 tests passed. [Full report and plot](REPORT.md), [raw metrics](results.json),
[paired contrasts](analysis.json), [frozen protocol](../../VARIANCE_STUDY.md).
