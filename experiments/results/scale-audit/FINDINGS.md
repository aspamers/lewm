# TwoRoom scale diagnosis

The 1,000-update sweep was stopped at the user's request. All four TwoRoom
runs completed; the first PushT run was interrupted. Existing checkpoints
and results are preserved. No training fix has been applied.

We audited both selected regularizers, seeds 31/32, at 500 and 1,000 updates.
For each checkpoint we used identical batches of training and validation clips,
crossing float32/bfloat16 inference with stored versus batch-computed BatchNorm
statistics. No gradients or optimizer updates were performed. Checkpoint state
was restored before each condition; BatchNorm mutations were not saved.

## Main observations

Representative seed-31 results at 500 updates on training clips:

| Regularizer | Normalization | Precision | Mean coordinate std |
|---|---|---|---:|
| SIGReg | stored statistics | float32 | 0.8986 |
| SIGReg | batch statistics | float32 | 0.8775 |
| SIGReg | batch statistics | bfloat16 | 0.8657 |
| Floor + correlation | stored statistics | float32 | 0.00657 |
| Floor + correlation | batch statistics | float32 | 0.00454 |
| Floor + correlation | batch statistics | bfloat16 | 0.08383 |

Both seeds and both data splits exhibit the same qualitative pattern.
BatchNorm batch statistics do not restore float32 spread, ruling out stale
running statistics as the sole explanation. Mixed precision greatly changes
the spread of the combined-method representation, whereas SIGReg is relatively
stable. This is consistent with rounding differences being amplified in an
already low-variation representation; it is not proof that all encoded signal
is meaningless or that precision is the sole cause of training failure.

The combined encoder is already low-variation before its projector: on
validation data, seed 31 has mean feature std about 0.000103 in float32,
versus SIGReg's 0.01368. Thus this is not only an output-scale choice.

At 1,000 updates, combined batch-statistics/float32 spread remains approximately
0.00614 and 0.00468 for seeds 31 and 32. Longer training at the frozen settings
did not resolve the issue. The trained floor is soft and remains substantially
unsatisfied even in mixed precision; it does not enforce a hard minimum of one.

## Additional implementation concern

The correlation denominator clamps variance to 1e-4. Below coordinate std 0.01,
the resulting quantity ceases to be ordinary scale-invariant correlation:
shrinking embeddings can shrink the measured correlation penalty. Nearly all
combined-method float32 coordinates fall in this regime at 500 updates.
This is a potential low-scale incentive, not yet an isolated causal finding.

## Implication

Do not interpret the modest planning-score lead as proof of superior learned
representations. First isolate precision and floor/correlation interactions in
short controlled tests. A float32 control, separate floor/correlation weighting,
and careful treatment of the correlation denominator are candidate investigations.
Changing several together or merely extending training would obscure the cause.

Raw results, including floor penalty, correlation penalty, pre-projector spread,
and fraction of coordinates below the correlation denominator threshold, are in
results.json. The executable audit is experiments/benchmark/scale_audit.py.
