# Radial shape sweep: beyond Gaussian toward a sphere

Protocol fixed before running. Five targets, coefficients 0.03/0.09/0.27 for
each, five paired training seeds 13 through 17, 1,000 updates: 75 runs.
Unchanged synthetic CNN/JEPA model (529,790 parameters, width D=192), optimizer,
one-step latent objective, batch size 64, 64 random projections, 17 frequency
knots and SIGReg integration window. No decoder or explicit radius loss.

Train data seed 12000 (256 episodes), validation 24000 (128 episodes).
For each target, choose the coefficient minimizing mean validation planning
distance over seeds. Then choose the primary target by the same validation
criterion, before opening fresh test seed 60000 (128 episodes, 24 fixed goals).
Evaluate every target at its selected coefficient. Report the primary candidate
comparison with Gaussian and all other comparisons as exploratory. The target
search is larger than Gaussian's coefficient search; independent test limits
selection optimism but does not equalize that total search budget.

## Family and exact projected targets

Let X=R*U, where U is uniform on the unit sphere in D dimensions and independent
of R. For v>0, R^2 ~ Gamma(k=D/(2v), scale=2v). For v=0, R=sqrt(D).
Every target has E[X]=0, E[R^2]=D and Cov(X)=I.
Var(R^2)=2Dv; squared-radius CV^2=2v/D. The parameter v is a variance multiplier,
not Student-t degrees of freedom. v=1 reproduces N(0,I) exactly.

| Kind | v | Gamma shape k | Target squared-radius CV^2 |
|---|---|---|---|
| radial_broad | 8 | 12 | 0.0833333 |
| gaussian | 1 | 96 | 0.0104167 |
| radial_half | 0.5 | 192 | 0.00520833 |
| radial_quarter | 0.25 | 384 | 0.00260417 |
| radial_sphere | 0 | infinity (fixed radius) | 0 |

The isotropic projection CF for v>0 is
1F1(k; D/2; -D*t^2/(4k)); for the sphere it is 0F1(D/2; -D*t^2/4).
Derivation: conditional on R, a unit projection has CF
0F1(D/2; -R^2*t^2/4). Integrate its power series using
E[(R^2)^n]=(D/k)^n*(k)_n to obtain 1F1. At k=D/2 it reduces to
exp(-t^2/2). The Gaussian code path retains the exact original fp32 buffer.
Other targets use SciPy float64 special functions once at initialization,
then store fp32 buffers. No target sampling occurs during optimization.
Uniform directions and covariance are preserved for the whole joint target;
we do not prescribe incompatible univariate marginals to every projection.

## Checks and analysis

Before training: Monte Carlo agreement with independent joint samples at D=192;
radius moments and projected variance; independent 2D Bessel identities;
exact Gaussian loss/gradient equality with SIGReg; finite gradients and dimension
validation. Quantify target CF differences under the existing frequency window.

Primary performance metric: mean goal distance. Secondary: success at distance
<0.12. Same five-step CEM budget as earlier studies (64 candidates, two rounds,
eight elites). Use paired training-seed contrasts and 20,000 bootstrap resamples
(RNG seed 20260912), with 95% percentile intervals. Five seeds are a small sample;
these intervals condition on the fixed data/goals and selected hyperparameters.
They exclude variation across datasets and selection uncertainty. No
multiple-comparison correction; exploratory rankings are not definitive optima.

Track raw R/sqrt(D) distributions and quantiles, E[R^2]/D, Var(R^2)/D^2,
scale-independent squared-radius CV^2, centered CV^2, mean-vector magnitude,
latent variance, effective rank and fixed projection tails. Adjacent frames and
random projections are not independent statistical replicates.

Crucial interpretation rule: a spherical target winning or tying does not
establish spherical learned embeddings. At D=192 its projections are close to
Gaussian. If actual radius distributions do not separate or approach targets,
this sweep is inconclusive about the optimal radial shape under the current
finite-projection loss. Adding a radius loss would be a separate experiment.

```powershell
$env:PYTHONPATH = 'src'
python experiments/radial_study.py
python experiments/report_radial_study.py outputs/radial-study-1000
```

SciPy is available via `pip install -e '.[radial]'` or the pilot requirements.
Full LeWM config also accepts these target names with default D=192; changing
latent width requires setting regularizer.latent_dim to match. The full published
dataset/model benchmark is not run here.

Mathematical references: [1F1 series](https://dlmf.nist.gov/13.2),
[0F1 definition and limit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp0f1.html),
[SciPy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html).
