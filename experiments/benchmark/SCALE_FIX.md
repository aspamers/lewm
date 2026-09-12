# Scale fix and controlled check

The 1,000-update benchmark was stopped to investigate low TwoRoom embedding
spread. Original screen code/results remain in git history through 5fae6a0;
the corrected regularizer is a methodological change and must not be silently
mixed into resumes of the original experiments.

## Code changes

* Nonconstant coordinates are normalized by their actual sample standard
  deviation for the correlation penalty. Exactly constant coordinates use a
  safe denominator of one; their centered values remain zero. The previous
  fixed 1e-4 variance clamp rewarded shrinking already small embeddings.
* All moment calculations, including the correlation matrix product, explicitly
  disable autocast. Merely casting inputs to float32 did not protect matmul.
* The variance floor, relative weight, prediction loss and architecture are
  unchanged. This soft floor is not a guarantee of minimum spread, and exact
  point collapse remains stationary. No new shape prior or scale anchor is added.
* New benchmark runs default to full float32 training with TF32 disabled and
  encoder activation checkpointing enabled. Precision is recorded in the run
  settings and training metadata. Bfloat16 remains an explicit diagnostic option.
  Source/setting checks prevent reusing the old sweep's output directories.

Tests cover invariance to shrinking correlated features, the absence of a
radial shrink gradient from the correlation term, finite gradients for constant
coordinates, and precision of the moment calculation under autocast.

## Controlled real-data trial

`scale_fix_trial.py` runs a 2x2 experiment: old/corrected regularizer crossed
with bfloat16/float32 training. Each arm starts from seed 31 with identical
250-update minibatches of 64 real TwoRoom clips. Weight stays at 0.1; no new
coefficient search. TF32 is disabled. All arms enable encoder activation
checkpointing to keep full-precision training within GPU memory, including
the mixed-precision controls. This recomputes activations; it does not alter
the architecture or effective batch size.

Every 50 updates, the same reserved validation batch is encoded under both
precisions and both BatchNorm modes. BatchNorm buffers are restored after
measurement. These probes are descriptive and do not change stopping time.
No planning test goals are used. Results concern the scale mechanism and
must not be presented as a new planning-performance benchmark.

The four-arm two-update smoke check fits in GPU memory and completes forward,
backward, checkpointed encoding, and all probe conditions. Full trial results
are written to outputs/scale-fix-trial.
