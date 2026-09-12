# Planning reevaluation after training-only encoder BN calibration

No model weights were retrained or changed. Only the encoder projection BatchNorm
running mean/variance were recalculated using the same 512 training clips per environment.
Predictor normalization remains unchanged. The original selected coefficients remain frozen.
Planning uses the same 16 goals, seeds and reduced CEM budget as before.

## Individual results

| Cohort | Environment | Regularizer | Weight | Seed | Original success | Calibrated success |
|---|---|---|---:|---:|---:|---:|
| original_500 | tworoom | gaussian | 0.03 | 31 | 18.75% | 56.25% |
| original_500 | tworoom | gaussian | 0.03 | 32 | 43.75% | 68.75% |
| original_500 | tworoom | variance_decorrelation | 0.1 | 31 | 37.50% | 12.50% |
| original_500 | tworoom | variance_decorrelation | 0.1 | 32 | 31.25% | 31.25% |
| original_500 | pusht | gaussian | 0.03 | 31 | 0.00% | 0.00% |
| original_500 | pusht | gaussian | 0.03 | 32 | 0.00% | 0.00% |
| original_500 | pusht | variance_decorrelation | 0.3 | 31 | 0.00% | 6.25% |
| original_500 | pusht | variance_decorrelation | 0.3 | 32 | 0.00% | 6.25% |
| original_1000 | tworoom | gaussian | 0.03 | 31 | 18.75% | 100.00% |
| original_1000 | tworoom | variance_decorrelation | 0.1 | 31 | 37.50% | 31.25% |
| original_1000 | tworoom | gaussian | 0.03 | 32 | 25.00% | 62.50% |
| original_1000 | tworoom | variance_decorrelation | 0.1 | 32 | 31.25% | 31.25% |
| corrected_fp32_250 | tworoom | variance_decorrelation | 0.1 | 31 | 25.00% | 31.25% |
| corrected_fp32_250 | tworoom | variance_decorrelation | 0.3 | 31 | 31.25% | 56.25% |
| corrected_fp32_250 | tworoom | variance_decorrelation | 1 | 31 | 18.75% | 43.75% |

## Means within matched cohorts

| Cohort | Environment | Regularizer | Weight | Seeds | Original mean | Calibrated mean |
|---|---|---|---:|---:|---:|---:|
| corrected_fp32_250 | tworoom | variance_decorrelation | 0.1 | 1 | 25.000% | 31.250% |
| corrected_fp32_250 | tworoom | variance_decorrelation | 0.3 | 1 | 31.250% | 56.250% |
| corrected_fp32_250 | tworoom | variance_decorrelation | 1 | 1 | 18.750% | 43.750% |
| original_1000 | tworoom | gaussian | 0.03 | 2 | 21.875% | 81.250% |
| original_1000 | tworoom | variance_decorrelation | 0.1 | 2 | 34.375% | 31.250% |
| original_500 | pusht | gaussian | 0.03 | 2 | 0.000% | 0.000% |
| original_500 | pusht | variance_decorrelation | 0.3 | 2 | 0.000% | 6.250% |
| original_500 | tworoom | gaussian | 0.03 | 2 | 31.250% | 62.500% |
| original_500 | tworoom | variance_decorrelation | 0.1 | 2 | 34.375% | 21.875% |

Random-policy scores on these same goals: TwoRoom 37.5%, PushT 0%.

Original 500/1,000-update cohorts used the old loss implementation and mixed precision.
The corrected 250-update cohort uses full precision and independent floor coefficients
with correlation coefficient 0.1. It currently has only seed 31 and is not a matched
comparison with the original Gaussian cohort. Its uncalibrated planning scores were
first measured during this reevaluation. The interrupted sweeps were not resumed.

These are exploratory reevaluations of repeatedly inspected goals. Calibration itself
uses training data only, but the broader diagnostic choices followed earlier outcomes.
Fresh independent confirmation is required for general performance claims.
