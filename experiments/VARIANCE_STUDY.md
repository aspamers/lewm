# Shape-free anti-collapse study

Frozen protocol: same 529,790-parameter CNN/JEPA, D=192, original one-step
latent prediction loss, no decoder. 1,000 updates, paired seeds 18 through 22.
Train 256 episodes seed 12000; validation 128 seed 24000; fresh final test
128 seed 72000, with 24 fixed nontrivial planning goals. This is a synthetic
pilot, not the published LeWM/TwoRoom benchmark.

## Four arms and equal active-arm search budgets

* None: coefficient 0, collapse control.
* Variance floor: coefficients 0.3, 3, 30.
* Variance floor plus decorrelation: coefficients 0.3, 3, 30.
* Gaussian SIGReg: coefficients 0.03, 0.09, 0.27.

Ten settings times five seeds = 50 training runs. Three settings per active
regularizer; identical numeric coefficients are not required because loss
scales differ. No floor threshold or internal decorrelation coefficient search.
Choose one coefficient per arm using mean validation planning distance before
opening test. Report all four arms; do not choose a new winner using test.
If a best coefficient is at a grid boundary, report that limitation rather than
claiming this implementation or regularizer family has been exhaustively tuned.

## Loss definitions

Compute statistics across batch separately at each time, using unbiased sample
variance. Floor = mean(ReLU(1-sqrt(variance+1e-4))^2).
No mean anchor, upper variance bound, shape matching or decorrelation is hidden
in the floor-only arm. It allows mixtures and perfectly redundant coordinates.

The combined arm adds mean squared off-diagonal sample correlation, normalized
by D*(D-1), at relative coefficient 1. Standardization clamps variance at 1e-4.
Correlation rather than covariance avoids directly rewarding small scales.
The floor term is identical between the two arms. This differs from the earlier
MomentRegularizer, which combined a mean anchor, floor, and unnormalized
off-diagonal covariance penalty; that earlier result is not a test of this arm.
Batch size 64 < D=192 means zero sample correlation is not jointly achievable
with all dimensions active. The penalty is a soft objective, not a hard demand.
Both floor variants have zero gradient at exact constant collapse; their role
is prevention from a random initialization, not guaranteed escape from symmetry.

Unchanged AdamW lr 5e-4, decay 1e-3, clipping 1, batch 64, matching initializations
and minibatches per seed. SIGReg retains 64 random projections and 17 frequencies.
The shared model's existing BatchNorm layers remain in every arm.

## Outcomes and interpretation

Primary comparison: floor-only versus SIGReg mean final planning distance.
Secondary: combined versus floor-only and SIGReg; all versus no regularizer.
Use paired seed differences with 20,000 seed bootstrap resamples (RNG 20260912)
and 95% percentile intervals. Five seeds are a small sample. Intervals condition
on selected coefficients and fixed tasks, excluding search and dataset
uncertainty. Secondary intervals are exploratory, without multiplicity correction.

Planning uses five-step CEM, 64 candidates, two iterations, eight elites, success
distance <0.12. Simulator states score plans and fit diagnostic probes only;
they are never training targets. Same goals per model, not independent new tasks.

Measure success, physical position probe MSE, five-step probe MSE, action-shuffle
degradation, raw and variance-normalized one-step latent MSE, latent variance,
effective covariance rank, effective correlation rank, largest correlation
eigenvalue share, off-diagonal correlations, coordinate std quantiles, fraction
of std below 0.1 and at least 0.9 (both pooled and per time), and mean drift.
Raw latent MSE alone can favor collapse or low scale; interpret it alongside
physical probes and planning. High coordinate variance alone does not establish
high-dimensional information. Relative effective rank alone does not exclude
near-zero absolute variance collapse either.

```powershell
$env:PYTHONPATH = 'src'
python experiments/variance_study.py
python experiments/report_variance_study.py outputs/variance-study-1000
```

Existing full-model config accepts target=variance_floor or
target=variance_decorrelation. Full benchmark training is separate work.
