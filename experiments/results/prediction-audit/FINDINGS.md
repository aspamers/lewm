# Prediction error audit: encoder projection BatchNorm statistics

The floor-weight sweep was stopped at the user's request. Three seed-31 arms
completed (floor weights 0.1, 0.3 and 1); their checkpoints are preserved. No
further training, planning sweep, or persistent checkpoint calibration was run.

## Controlled comparison

Six old and new checkpoints were measured on the same 256 validation clips.
We independently changed encoder projection BatchNorm, predictor projection
BatchNorm, predictor dropout (including functional attention dropout), and
precision. Each condition reloaded the original checkpoint. Calibration uses
population moments from 512 separate training clips, without gradients or
weight updates. The calibrated buffers exist only in audit memory.

| Checkpoint | Original eval MSE | Encoder BN calibrated MSE | Calibrated normalized MSE | Persistence normalized MSE |
|---|---:|---:|---:|---:|
| Original Gaussian, 500 updates | 162.067 | 0.04601 | 0.052 | 0.259 |
| Original combined, 500 updates | 3.985 | 0.00070 | 33.419 | 1.298 |
| Legacy loss, float32, 250 updates | 21.898 | 0.00797 | 1.025 | 1.688 |
| Corrected loss, float32, 250 updates | 22.003 | 0.00797 | 1.025 | 1.688 |
| Corrected, floor 0.3, 250 updates | 4520.137 | 0.03336 | 0.069 | 0.352 |
| Corrected, floor 1, 250 updates | 310.762 | 0.04159 | 0.050 | 0.316 |

The huge errors predate the fixes, including in the Gaussian baseline. Changing
predictor BatchNorm alone or turning dropout on does not resolve them. Using
batch-computed statistics for the encoder's projection BatchNorm, or replacing
that layer's running statistics with training-data moments, does resolve the
large absolute-error spike. Recalibrating predictor BatchNorm in addition gives
little further benefit here. The component is the projection MLP's BatchNorm,
not a BatchNorm layer inside the ViT.

## What this establishes

* The apparent regression after the fixes was largely a newly exposed,
  pre-existing inference-normalization mismatch, not evidence that the fixes
  introduced the large evaluation errors.
* Precision-induced near-collapse remains a separate problem: the old combined
  model still has poor normalized prediction error after calibration. Its tiny
  absolute error cannot be interpreted as good prediction because targets also
  have almost no variation.
* At floor weight 1, full-precision corrected training plus encoder-statistic
  calibration yields evaluation std approximately 0.913 and normalized prediction
  MSE approximately 0.050, compared with persistence 0.316. These are encouraging
  diagnostics of spread and short-horizon prediction, not planning success.
* Weight 0.3 also improves normalized prediction relative to the 0.1 control.
  Only one seed has completed the stronger-floor arms. This audit cannot choose
  a robust optimum or establish an advantage over Gaussian.

## Consequences for further experiments

Any resumed comparison needs a documented training-only encoder BatchNorm
calibration step (or a separately tested architecture change) before evaluation,
applied equally to both methods. Preserve original results rather than replacing
them silently. Earlier planning scores were measured with the normalization
mismatch and should not establish an algorithmic ranking. No new planning scores
were obtained here, and undertraining is no longer a sufficient explanation of
the earlier weak baseline without first addressing the evaluation mismatch.

Source code: experiments/benchmark/prediction_audit.py. Raw condition results
and checkpoint hashes are retained in this directory.
