# Covariance-matched Student-t targets

Question: does relaxing Gaussian tails help while retaining a single central
mode and the same target mean and covariance?

Completed [36-run findings](results/student-t-study-1000/FINDINGS.md) include a
fresh final test set, all validation settings and learned-versus-target tail plots.

Compare Gaussian SIGReg with Student-t degrees of freedom **3, 5, 9**, each
standardized to covariance identity. Every arm uses the same model, latent width
192, one-step prediction loss and optimizer; no decoder is present.

## Target math

For `G ~ N(0,I)` and `U ~ chi-square(nu)`, define
`X = G * sqrt((nu-2)/U)`, with one shared radial multiplier per vector.
Then `Cov(X)=I` for `nu>2`. All unit projections follow the same variance-one
Student-t. Independent t samples per coordinate would be a different joint target.

With `a=sqrt(nu-2)*abs(t)`, the target characteristic functions are:

- df 3: `(1+a) exp(-a)`
- df 5: `(1+a+a^2/3) exp(-a)`
- df 9: `(1+a+3a^2/7+2a^3/21+a^4/105) exp(-a)`

These are cached once. The projection sampler, frequency grid and integration
weights—including SIGReg's Gaussian frequency window—are unchanged. This
isolates the target distribution, not a different estimator. Unit variance and
agreement with independently sampled multivariate t distributions are tested.

## Pilot

36 training runs: four targets × three weights × three initialization seeds.
Weights are 0.03, 0.09, 0.27; each run has 1,000 updates. The architecture and
optimization match the direct regularizer pilot: 529,790 parameters, a small
CNN, 192-dimensional latent, BatchNorm projectors and a reduced predictor.
This is not the full LeWM/TwoRoom benchmark.

Training and validation data retain seeds 12000 and 24000. **Final test seed is
48000**, because the previous study's test data have already been inspected.
Gaussian is rerun for a paired comparison on this new set. One coefficient per
target and the primary Student-t df are chosen using mean validation planning
distance, before the final test evaluation. All four targets are reported.

The Student-t family has nine df/weight candidates versus Gaussian's three;
reporting the independent test limits, but does not eliminate, comparison issues
from unequal search budgets. Twenty-four test goals and three model seeds provide
only a small feasibility check, not a definitive ranking.

```powershell
$python = 'C:\Users\aspam\.cache\lewm-experiments\venv\Scripts\python.exe'
$env:PYTHONPATH = 'src'
& $python -m pytest tests -q
& $python experiments/student_t_study.py --steps 1000 --seeds 0 1 2 --output outputs/student-t-study-1000
& $python experiments/report_student_t.py outputs/student-t-study-1000
```

Dependencies: `requirements-pilot.txt`, plus matplotlib 3.11.2 for reporting.
Use a fresh output directory to retain prior runs. Checkpoints remain in ignored
`outputs/`; compact reports and raw metrics can be committed separately.

## Full LeWM model

The existing full-model config accepts these targets without any architecture
changes:

```bash
python train.py --config-name=regularizer_study data=tworoom target=student_t5 experiment.sigreg_weight=0.09
# Print the full sweep without running it:
python experiments/student_t_matrix.py --data tworoom
```

That path retains the original ViT, predictor and four-frame objective, plus the
episode-disjoint split. It requires upstream dependencies and published datasets;
full benchmark training has not been run here. Do not assume the small pilot's
selected coefficient transfers unchanged.

References: [SciPy multivariate-t parameterization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html),
[NIST explicit Bessel formulas](https://dlmf.nist.gov/10.49).
