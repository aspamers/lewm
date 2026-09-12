# TwoRoom and PushT: bounded real-environment screening

Gaussian SIGReg versus variance floor plus decorrelation. These are short
runs of the full LeWM architecture, not the published 100-epoch schedule.

## Held-out planning success

Selection uses mean validation success across the two model seeds. Fixed
coefficients (SIGReg 0.09, floor + decorrelation 0.3) were transferred
from the synthetic pilot. Test goals were reserved before training.

| Environment | Setting | Regularizer | Weight | Seed 31 | Seed 32 | Mean |
|---|---|---|---:|---:|---:|---:|
| tworoom | selected | gaussian | 0.03 | 18.75% | 43.75% | 31.25% |
| tworoom | selected | variance_decorrelation | 0.1 | 37.50% | 31.25% | 34.38% |
| tworoom | fixed | gaussian | 0.09 | 25.00% | 25.00% | 25.00% |
| tworoom | fixed | variance_decorrelation | 0.3 | 31.25% | 25.00% | 28.12% |
| pusht | selected | gaussian | 0.03 | 0.00% | 0.00% | 0.00% |
| pusht | selected | variance_decorrelation | 0.3 | 0.00% | 0.00% | 0.00% |
| pusht | fixed | gaussian | 0.09 | 0.00% | 0.00% | 0.00% |
| pusht | fixed | variance_decorrelation | 0.3 | 0.00% | 0.00% | 0.00% |

* tworoom: random-policy success 37.50% on the same test goals.
* pusht: random-policy success 0.00% on the same test goals.

## Training cost

Wall time includes data loading, optimization, progress output and checkpoints.
Runs were sequential in a fixed order, so timing is descriptive rather than
a randomized systems benchmark. Peak allocation is PyTorch's allocated memory.

| Environment | Regularizer | Median seconds / run | Median peak GiB |
|---|---|---:|---:|
| tworoom | gaussian | 91.7 | 6.642 |
| tworoom | variance_decorrelation | 94.0 | 6.642 |
| pusht | gaussian | 95.1 | 6.642 |
| pusht | variance_decorrelation | 94.3 | 6.642 |

Isolated forward/backward timing uses identical (4, 64, 192) bf16
embeddings, 40 warmups and 200 alternating-order measurements per arm.

| Regularizer | Median GPU ms | Median wall ms |
|---|---:|---:|
| gaussian | 1.122 | 1.151 |
| variance_decorrelation | 1.179 | 1.216 |

## Embedding spread

Computed from the same 512 reserved test clips for every checkpoint,
using one frame per clip. Rank must be interpreted alongside absolute spread.

| Environment | Regularizer | Weight | Seed | Mean std | Entropy rank | Mean absolute correlation |
|---|---|---:|---:|---:|---:|---:|
| tworoom | gaussian | 0.03 | 31 | 0.9150 | 2.11 | 0.622 |
| tworoom | gaussian | 0.03 | 32 | 0.9026 | 2.55 | 0.546 |
| tworoom | gaussian | 0.09 | 31 | 0.9123 | 5.80 | 0.359 |
| tworoom | gaussian | 0.09 | 32 | 0.8521 | 6.31 | 0.348 |
| tworoom | variance_decorrelation | 0.1 | 31 | 0.0066 | 8.19 | 0.305 |
| tworoom | variance_decorrelation | 0.1 | 32 | 0.0054 | 8.60 | 0.318 |
| tworoom | variance_decorrelation | 0.3 | 31 | 0.0541 | 8.21 | 0.316 |
| tworoom | variance_decorrelation | 0.3 | 32 | 0.0496 | 9.02 | 0.304 |
| pusht | gaussian | 0.03 | 31 | 0.8079 | 2.80 | 0.507 |
| pusht | gaussian | 0.03 | 32 | 0.8447 | 3.02 | 0.497 |
| pusht | gaussian | 0.09 | 31 | 0.8745 | 7.06 | 0.333 |
| pusht | gaussian | 0.09 | 32 | 0.8594 | 7.37 | 0.320 |
| pusht | variance_decorrelation | 0.3 | 31 | 0.5895 | 5.05 | 0.388 |
| pusht | variance_decorrelation | 0.3 | 32 | 0.6089 | 3.27 | 0.495 |

## Interpretation limits

* Two model seeds and 16 goals per split provide screening evidence only. Paired
  model seeds reuse goals; 32 outcomes are not 32 independent environments.
* Each run receives 500 updates at batch 64. The CEM budget is reduced to
  64 candidates and three iterations. Poor scores can reflect undertraining
  or insufficient planning search, not an inherent limit of the regularizer.
* Compare learned-policy success with the random control before attributing
  value to either learned representation. Equal success does not prove equivalence.
* Only coefficients transfer from the synthetic pilot; trained model weights
  do not transfer between environments. Cube and Reacher remain untested.
* Original archives omit evaluation seeds. Deterministic reset seeds are
  generated, then recorded start and goal states are restored. Per-environment
  data-check.json records rendered-image and one-step simulator discrepancies.

Protocol, source hashes, partitions, validation scores, per-goal test outcomes
and timing are retained beside this report. No test results guide selection.
