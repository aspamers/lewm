# Scale-fix trial: precision drives the observed failure

Four paired fresh TwoRoom runs completed: one seed, 250 updates, batch 64,
weight 0.1. Old versus corrected correlation loss was crossed with bfloat16
versus float32 training. All arms used the same initialization and clip sequence,
encoder activation checkpointing, and TF32 disabled. No final test goals were
used; spread probes use one fixed validation batch. See manifest and curves.

| Loss | Training precision | Float32 eval std | Float32 batch-statistics std |
|---|---|---:|---:|
| Old | bfloat16 | 0.00530 | 0.00328 |
| Corrected | bfloat16 | 0.00499 | 0.00329 |
| Old | float32 | 0.16272 | 0.08701 |
| Corrected | float32 | 0.16286 | 0.08702 |

The correction alone does not fix the observed mixed-precision failure.
Full-precision training increases true evaluation spread by approximately 31x
and batch-statistics spread by approximately 27x relative to the old mixed-
precision control. Both float32 arms behave nearly identically here: the old
correlation clamp is not the dominant cause of this run's failure once precision
is corrected. Its low-scale shrink incentive remains a mathematical defect,
covered by regression tests and removed in the corrected implementation.

New benchmark training defaults to float32 with encoder activation checkpointing
and TF32 disabled. The regularizer explicitly computes all moments in float32.
Old numerical results and paused checkpoints retain their original meaning;
they must not be resumed with the changed method under the original protocol.

## Remaining limitations

* A 0.087 batch spread remains well below the soft floor target of one. The
  floor penalty and prediction loss trade off at the current coefficient; a
  corrected implementation does not turn a soft penalty into a hard constraint.
* The eval/batch spread difference remains visible. Calibration of BatchNorm
  statistics and regularization strength can be studied separately if needed.
* Float32 is a numerical safeguard, not a free optimization: these trial runs
  took about 101 seconds versus 55–56 seconds for mixed precision with the same
  activation-checkpointing setting. This is not a systems benchmark.
* One seed and a short budget isolate a mechanism; they do not establish
  robust planning gains or superiority over SIGReg/Sub-JEPA. No new planning
  performance claim follows from these results.

The broad environment sweep remains stopped. Any new performance comparison
must record the corrected method and matched precision budgets explicitly.
