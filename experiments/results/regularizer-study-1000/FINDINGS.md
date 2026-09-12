# Findings: can a mixture replace the Gaussian target?

**Yes, in this small synthetic study a fixed two-Gaussian target avoided collapse,
formed two modes, retained position information and supported action-conditioned
planning. It did not improve average planning performance over original SIGReg.**

We trained 39 models: three seeds across a predefined coefficient sweep, with
the same 192-dimensional latent, architecture, one-step prediction objective,
data and optimization settings. No decoder or reconstruction loss was used.
Coefficients were selected using validation planning, before a separate test set.

| Regularizer | Mean test planning success |
|---|---:|
| Original SIGReg target | 81.9% |
| Fixed two-Gaussian mixture | 65.3% |
| Single Gaussian with the mixture's covariance | 59.7% |
| Moment constraints | 56.9% |
| No regularizer | 1.4% |

Each mean is across three model seeds on the same 24 test goals. Variability is
substantial: the full report contains seed standard deviations, distances,
probes and raw per-episode results. These are not published TwoRoom benchmark
scores; the model uses a small CNN and reduced predictor capacity.

The mixture result is not just a distribution label. Its histogram has two
peaks: 74.2% of test embeddings lie near the prescribed centers along the mixture
axis, and only 5.5% lie in the central interval. Position-probe MSE is 0.00142,
versus 0.00102 for SIGReg and 0.267 for no regularizer. Shuffling actions worsens
the mixture's future-position predictions, showing that its dynamics use actions.

The covariance control matters. The mixture's effective rank is 2.89, versus
14.12 for SIGReg, but the matched single Gaussian is also low-rank at 2.42.
Therefore low effective rank is not evidence that multimodality uniquely allows
compact information storage. Both alternative targets prescribe the same strongly
correlated covariance, which also changes Euclidean goal-cost geometry.

Two limits keep the interpretation narrow. A fixed two-component mixture is a
different prescribed distribution, not a fully flexible family permitting any
mixture. And the moment arm tests one soft penalty with untuned internal weights;
its result does not rule out better moment-based restrictions.

The useful conclusion is **feasibility of a non-Gaussian target, without evidence
of an improvement from this particular target**. A broader mixture family and
tests with identity aggregate covariance remain separate experiments.

[Full protocol and measurements](REPORT.md) · [Raw results](results.json)

![Test latent distributions](mixture-axis.png)
