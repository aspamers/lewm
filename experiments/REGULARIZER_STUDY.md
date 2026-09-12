# Direct regularizer study

This returns to the original question: is LeWM's single-Gaussian restriction
necessary, or can a non-Gaussian / less prescriptive alternative work?

The completed [39-run findings](results/regularizer-study-1000/FINDINGS.md)
include separate test evaluation, all validation coefficients, and latent histograms.

Unlike the earlier decoder pilot, this study has **no decoder**, a fixed latent
width of 192, and the unchanged one-step latent prediction loss with coefficient
one. Only the regularizer and its coefficient vary.

## Arms

1. Original SIGReg: standard Gaussian target.
2. Fixed symmetric two-component Gaussian mixture.
3. A single Gaussian with exactly the mixture's covariance.
4. Mean, standard-deviation-floor and off-diagonal-covariance penalties, without
   a distribution-shape target.
5. No regularizer.

The mixture is `0.5 N(+m, 0.25 I) + 0.5 N(-m, 0.25 I)`, where each coordinate of
`m` is `sqrt(0.75)`. It has zero mean and unit **marginal** variance, but its
covariance is `0.25 I + m m^T`, not identity. The third arm controls for this
covariance change so that multimodality is not confused with covariance geometry.

For a unit direction `a`, the mixture characteristic function is
`exp(-0.5 * 0.25 * t^2) * cos(t * a^T m)`. The covariance-matched Gaussian uses
`exp(-0.5 * t^2 * (0.25 + (a^T m)^2))`. These are projections of a single coherent
multivariate distribution, not independently fitted one-dimensional mixtures.
The projection sampler, frequency grid and quadrature weights are those of SIGReg.

A fixed mixture prescribes another shape; it is not mathematically a more
permissive superset of Gaussian matching. The moment arm is the broader family
of shapes. This first study does not fit a flexible mixture family, learn its
centers, or search the number of components or their separation.

## Executed pilot protocol

The synthetic world and small CNN are reused solely to keep the first study
tractable locally. Every arm uses the same 529,790-parameter model, fixed
192-dimensional latent, BatchNorm projectors, and a two-block hidden-64
predictor. This is not the published LeWM architecture or TwoRoom benchmark.

- 39 training runs: three initialization seeds across thirteen configurations.
- Three CF weights (0.03, 0.09, 0.27), three moment weights (0.3, 3, 30), and zero
  weight for the no-regularizer control. Numeric scales differ between objectives.
- 1,000 updates, batch 64, AdamW lr 5e-4, weight decay 1e-3, clipping 1.
- Identical initialization and minibatch schedules across arms within each seed.
- Separate training, validation and final-test episodes, with fixed data seeds.
- Choose one weight per regularizer using mean validation planning distance,
  then evaluate on the final test set without revising that choice.
- Same latent-distance CEM planner and budget for all models: horizon five,
  64 candidates, two iterations, eight elites; 24 nontrivial goals per split.
- Simulator state is used for scoring and fitted diagnostic probes only.

Inspect planning distance and success, position probes, absolute latent variance,
effective rank, action-shuffle sensitivity and actual mixture-axis histograms.
Low latent prediction loss alone is not a success criterion.

```powershell
$python = 'C:\Users\aspam\.cache\lewm-experiments\venv\Scripts\python.exe'
$env:PYTHONPATH = 'src'
& $python -m pytest tests -q
& $python experiments/regularizer_study.py --steps 1000 --seeds 0 1 2 --output outputs/regularizer-study-1000
& $python experiments/report_regularizer_study.py outputs/regularizer-study-1000
```

Use `requirements-pilot.txt`; reporting also requires matplotlib (executed with
version 3.11.2). Checkpoints stay under ignored `outputs/`; compact results and
figures are committed separately. An existing directory is overwritten on rerun,
so use a fresh output path to retain older results.

## Full model configuration

`config/train/regularizer_study.yaml` retains upstream's ViT, original predictor,
192-dimensional width and four-frame training window. It uses the experiment
episode split and training-only action normalization already added to this fork.

```bash
python train.py --config-name=regularizer_study data=tworoom target=mixture
python train.py --config-name=regularizer_study data=tworoom target=moments experiment.sigreg_weight=0.3
python train.py --config-name=regularizer_study data=tworoom target=none experiment.sigreg_weight=0
# Print the full sweep, without launching it:
python experiments/regularizer_matrix.py --data tworoom
```

Full benchmark training requires the upstream dependencies and datasets and has
not been run here. Coefficients need validation again at the full model's batch
size and budget. The small pilot tests feasibility and failure modes, not a
general claim that one regularizer is superior.
