# Predictive bottleneck experiment

Fork of Mengarr/lewm at `8a2c595813d0eee85b2dbffa6f58ff0842f9e673`.

Test whether future-observation supervision can preserve useful information
without SIGReg. This changes the objective, not dense-network compute. No gates,
whitening, rank floor, EMA, or pruning are introduced.

## Implemented comparison

The benchmark manifest includes a reference-width model, compact LeWM at widths
16 and 32, compact observation predictors with and without SIGReg, and a compact
LeWM control using the same reduced latent-loss coefficient as the observation
models. The extra control separates the beta change from decoder supervision.

`BottleneckPredictor` projects latents into a 192-wide transformer. The action
encoder remains 192-wide, with transformer depth, attention heads and MLP widths
unchanged. Encoder and predictor projectors map 192 to the selected latent width.
The small boundary projections change parameter counts slightly.

The latent term retains upstream's one-step teacher-forced loss over the
history. The additional observation term rolls out without future observations
and scores horizons 1, 3, 5 equally, using MSE at scales 1, 2, 4. Targets are actual
preprocessed future images, so the decoder has unconstrained outputs rather than
a tanh incompatible with ImageNet-normalized pixels. `beta=0.1`; optional SIGReg
weight is 0.09. The SIGReg term sees the same eight-frame windows across all
matched variants. This matched protocol is distinct from the untouched upstream
four-frame recipe.

Experiment training splits entire episodes with fixed split seed 42, fits action
statistics on training episodes only, and records both. Simulator state is never
a training target. Model seeds and the episode split seed are separate.

## Local synthetic pilot

This requires only PyTorch and the packages in `requirements-pilot.txt`.
It uses a small CNN and synthetic spot/wall dynamics, **not the published
TwoRoom environment**, and reduced transformer capacity. It checks learning
behavior before the expensive benchmark runs.

On this Windows machine the functioning environment is outside OneDrive at
`C:\Users\aspam\.cache\lewm-experiments\venv`. OneDrive interfered with installing
packages into the workspace `.venv`; use the external environment below.

```powershell
$python = 'C:\Users\aspam\.cache\lewm-experiments\venv\Scripts\python.exe'
$env:PYTHONPATH = 'src'
& $python -m pytest tests -q
& $python experiments/pilot.py --steps 1000 --seeds 0 1 2 --output outputs/pilot-1000
& $python experiments/analyze_pilot.py outputs/pilot-1000
```

To create a fresh CUDA environment, use Python 3.12 and install the pinned
`experiments/requirements-pilot.txt` with `uv pip install --link-mode copy`.
The pilot writes per-run weights and JSON under `outputs/`, which Git ignores.
The compact baseline's decoder is optimized from detached forecast latents;
it does not supply representation gradients. Decoder gradients are clipped
separately so they cannot affect baseline gradient clipping.

The baseline has latent weight 1; `compact_low_beta`, `predictive`, and
`predictive_sigreg` have latent weight 0.1. The two latter variants therefore
differ only by SIGReg. All use equal minibatch schedules and independent held-out
episodes. Fixed data seeds are 12000 and 24000. SIGReg uses 64 projections for this
pilot (the full config uses the upstream 1024).

## Full benchmark launch

Install the full upstream dependencies from `pyproject.toml`/`uv.lock` in a
separate environment and obtain the datasets linked in the upstream README.
Full training and simulator evaluation have not yet been executed here. The
Hydra model configuration has been composed and its forward shapes tested.

```bash
# Prints 27 commands: 9 configurations x 3 seeds; no training by default.
python experiments/matrix.py --data tworoom
# Execute sequentially once dependencies and tworoom.h5 are available.
python experiments/matrix.py --data tworoom --run
```

Use equal data, epochs, batch size, optimizer, and planner budget across variants.
Start with TwoRoom, then extend to PushT and the configured manipulation/control
tasks. Keep final planning episodes separate from model-selection trajectories.
The existing unmodified recipe remains available with `python train.py data=tworoom`.

For a decoder-equipped checkpoint, use `eval.py` with the default latent goal cost
for the main comparison. Add `+planning_cost=decoded` for the diagnostic cost to
the real terminal goal image. Supply `+action_stats=/absolute/path/to/split_and_action_stats.json`
for the training run's action normalization. Reuse identical evaluation seed and
planner settings for both costs. The full benchmark baselines do not train a
decoder; fit a frozen diagnostic decoder before comparing their observation error.

## Decision criteria and remaining work

Primary endpoint: planning success with the existing latent goal cost. Secondary:
held-out future-image error per horizon, physical-state probes, latent spectrum
and absolute variance, action-shuffle and feature-ablation sensitivity. Record
training seconds, memory and parameter counts separately from planning cost.

Prediction improvement with gains only under decoded planning suggests a goal
metric problem. Low pixel error with poor probes/action sensitivity suggests a
background shortcut. Do not infer successful information selection from effective
rank alone when absolute variance approaches zero. Adaptive gates should wait
until an observation objective reliably preserves useful dynamics.

Still outstanding: published-data training, benchmark planning success,
contact/rare-event stratification, and full-model frozen diagnostic decoders.
