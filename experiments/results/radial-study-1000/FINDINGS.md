# Radial sweep: no clear optimum, weak control of actual shape

Completed 75 synthetic training runs: five radial targets, three coefficients
each, five paired seeds, 1,000 updates. All targets have theoretical mean zero
and covariance I. Only SIGReg's projected target changes. No decoder, direct
radius penalty, or architecture change was added.

## Performance

| Target | Validation-selected weight | Fresh-test distance (lower better) | Success |
|---|---|---|---|
| Broad radial distribution | 0.09 | 0.08653 | 75.8% |
| Gaussian | 0.03 | 0.08530 | 79.2% |
| Half Gaussian squared-radius variance | 0.09 | 0.08459 | 78.3% |
| Quarter Gaussian squared-radius variance | 0.09 | 0.08424 | 77.5% |
| Sphere surface | 0.09 | 0.08478 | 76.7% |

The broad target won validation, but its fresh-test distance was 0.00123 worse
than Gaussian; paired five-seed bootstrap 95% interval [-0.00711, +0.00836].
Every alternative-versus-Gaussian distance interval includes zero. The quarter
target has the lowest numerical test distance, but this does not establish an
optimum. Test rankings were not used to select a new primary candidate.

The Gaussian coefficient changed from earlier experiments because validation
was repeated with new model seeds. This test split is also new (60000), so raw
scores should not be compared directly with the earlier Student-t tables.

## The more informative finding: targets were poorly realized

At a common coefficient of 0.09, validation squared-radius CV^2 was:

| Target | Intended CV^2 | Actual CV^2 |
|---|---|---|
| Broad | 0.083333 | 0.314144 |
| Gaussian | 0.010417 | 0.264571 |
| Half | 0.005208 | 0.261125 |
| Quarter | 0.002604 | 0.259136 |
| Sphere | 0 | 0.257570 |

This scale-independent metric measures variation in squared embedding length.
Switching the target from Gaussian to sphere reduces it by only 2.65%, far
short of the intended reduction to zero. Even the strongest tested coefficient
(0.27) leaves the sphere's validation CV^2 at 0.121780. The sphere target did
not make the embeddings spherical. The Gaussian target did not make the full
joint embedding distribution Gaussian either.

In 192 dimensions the Gaussian and sphere projected characteristic functions
differ by at most 0.00283 on the existing grid. Their target-only weighted
squared difference is just 0.000205 after multiplying by batch size 64.
High-precision moment calculations confirm these small differences are real,
not a special-function numerical error. The full loss is still a finite-sample,
finite-projection soft penalty competing with prediction.

## What this establishes

This sweep found no clear preferred radial target under the current training
recipe. It does not establish that exact Gaussian, spherical, and intermediate
embedding distributions work equally well, because the learned distributions
did not approach those prescribed radial shapes.

A more discriminating follow-up would keep the angular constraint fixed and
directly match the radius distribution, verifying that the desired spread is
actually obtained before comparing planning. That follow-up has not been run.

Five seeds and 24 reused synthetic test goals limit inference. Bootstrap
intervals describe training-seed variability conditional on these data and
selected coefficients, not variation across tasks or tuning uncertainty.
This is not the full published LeWM/TwoRoom benchmark.

All 39 tests passed. [Detailed report and plots](REPORT.md), [raw metrics](results.json),
[derived comparisons](analysis.json), [frozen protocol](../../RADIAL_STUDY.md).
