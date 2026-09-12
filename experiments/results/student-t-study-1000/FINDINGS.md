# Findings: Student-t tails versus Gaussian

**Student-t worked as a regularizer, but there is no clear improvement over
Gaussian SIGReg in this pilot. The mildest tested tail relaxation, df=9, was
closest to Gaussian in planning and in the learned distribution.**

We ran 36 models: Gaussian and Student-t df 3/5/9, three loss coefficients each,
and three initialization seeds. Model, 192-dimensional latent, prediction loss,
optimizer and planner stayed fixed. There was no decoder. Every target had mean
zero and covariance identity; only its higher-order distribution shape changed.

| Target | Mean test success | Mean goal distance (lower is better) |
|---|---:|---:|
| Gaussian / SIGReg | 75.0% | 0.08874 |
| Student-t, df=3 | 72.2% | 0.09167 |
| Student-t, df=5 | 65.3% | 0.1026 |
| Student-t, df=9 | 77.8% | 0.09009 |

Validation selected df=9 as the primary Student-t candidate before final testing.
All four targets selected regularization weight 0.09. The df=9 model has slightly
higher thresholded success but slightly worse average distance than Gaussian.
Across three model seeds and 24 shared test goals, that is not convincing evidence
of superiority. The full report includes per-seed variation and raw per-goal distances.

The test data use fresh seed 48000, replacing the already-inspected test split
from the mixture study. Gaussian was rerun on this same set. These percentages
must not be compared directly with the earlier mixture-study percentages.

The distribution diagnostics also qualify the result. The very heavy df=3
target matched the center reasonably well but substantially underrepresented
its extreme tails. Actual latent variance was 127.7, versus target trace 192
and Gaussian's measured 196.4. Thus target covariance was controlled, but learned
covariance was not identical. Finite-budget characteristic-function matching did
not produce an exact Student-t representation.

This leaves a narrow conclusion: a modest tail relaxation is viable and roughly
competitive here; we have not found a better fixed target. Stronger tails did not
provide a reliable gain. This does not prove Gaussianity is essential, identify
the universally best shape, or settle what a simpler shape-free constraint can do.

Scope remains a small CNN/reduced-predictor synthetic pilot, not full LeWM or the
published TwoRoom benchmark. No meaningful compute saving is implied by changing
the cached target characteristic function.

[Full report and protocol](REPORT.md) · [All measurements](results.json)

![Learned projected distributions versus their targets](student-t-tails.png)
