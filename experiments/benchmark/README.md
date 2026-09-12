# Real-environment screening: TwoRoom and PushT

Scope narrowed by the user to the two simplest target environments. Cube and
Reacher are deferred. This is a bounded first pass on official datasets and
real simulators, not the earlier synthetic moving-spot data and not a reproduction
of the full 100-epoch published training schedule.

## Protocol

* Full architecture from `config/train/model/lewm.yaml`: ViT-tiny, patch 14,
  224-pixel images, 192-dimensional embeddings, six-block predictor.
* Two regularizers: Gaussian SIGReg and variance floor + correlation penalty.
* SIGReg weights 0.03/0.09/0.27; combined weights 0.03/0.1/0.3. Equal three-setting
  search budget, two paired model seeds 31/32, two environments: 24 training runs.
* Bounded budget after GPU smoke tests: 500 updates, batch 64, constant AdamW learning rate
  5e-5 and weight decay 1e-3, clipping 1, bf16 mixed precision. Runtime smoke
  tests determine whether the batch fits before scientific runs start. All
  candidates receive the same budget; no adaptive stopping based on results.
* Preserve the original four-frame one-step latent prediction objective,
  frameskip 5, no decoder, 1,024 SIGReg projections and 17 frequencies.
* Use a lossless per-frame HDF5 cache and whole-frame reads to avoid decoding
  100-frame archive chunks and slow strided HDF5 selections. Pixel values and
  action/episode metadata are preserved; evaluation uses the original archive.
* Isolate regularizer RNG from model dropout. Pair initializations and the full
  minibatch sequence across candidate settings. Save optimizer and RNG state
  every 100 updates for interruption recovery.
* Fixed episode partition: 80% training / 10% validation / 10% final test, seed
  9031, eligible episodes at least 30 frames long. Normalize actions using only
  training episodes. Do not sample evaluation starts from training episodes.
* Sixteen predefined start/goal pairs per validation/test split; goal offset 25
  raw steps. If a split has fewer than 16 episodes, some episodes repeat with
  separately sampled starts and must not be treated as independent trials.
* Shared reduced CEM budget: 64 candidates, three iterations, eight elites,
  horizon five action blocks, five raw actions per block, replanning every
  five blocks, episode evaluation budget 50 raw steps. This is cheaper than
  the repository default of 300 candidates and 30 CEM iterations.
* Select each arm's coefficient by mean validation success across seeds;
  ties follow the ascending predefined grid. Freeze selection before test.
* Evaluate both the selected coefficients and predeclared transferred settings
  (SIGReg 0.09, combined 0.3) on test. These fixed settings come from the earlier
  synthetic pilot, not the test outcomes. This tests hyperparameter transfer,
  not transfer of one model's weights between environments.
* Run a random-policy control on the same test goals. Save per-goal successes,
  training/evaluation time and peak GPU allocation. Two seeds and 16 goals are
  screening evidence only; do not claim equivalence or statistical superiority.

The official TwoRoom archive omits simulator seeds. Assign deterministic reset
seeds 6173 + episode index, then restore the recorded state/goal. This is explicitly
marked as generated evaluation seeds, not original data. Check recorded versus
restored rendered images and one-step state dynamics before training to detect
simulator/data mismatches. Original seeds are retained if present in a dataset.

## Reproducible local execution

The Docker base is pinned by digest; PyTorch 2.6.0 / torchvision 0.21.0 are shared
by both arms. This differs from the earlier native synthetic-pilot runtime.
The actual environment versions are saved before launch.

```powershell
docker build -t lewm-benchmark:20260912 -f experiments/benchmark/Dockerfile experiments/benchmark
python experiments/benchmark/download_data.py --root C:/Users/aspam/.cache/lewm-benchmark --environments tworoom pusht
```

Mount this repository at `/workspace` and the cache root at `/data`, enable GPU
access, then run `python experiments/benchmark/smoke.py` and
`python experiments/benchmark/run.py`. `manifest.json` records settings and
source hashes and prevents mixed-protocol resumes. `status.json` and individual
`progress.json` files record live progress. Training waits for checksum-verified
data; download failure stops the run.

`download_data.py` pins the official Hugging Face repository revisions and
verifies published SHA256 archive hashes. Only HDF5 payloads are extracted to
fixed destinations outside OneDrive. Preserve a 30-GiB free-disk reserve.
Official sources: [TwoRoom](https://huggingface.co/datasets/quentinll/lewm-tworooms)
and [PushT](https://huggingface.co/datasets/quentinll/lewm-pusht).

## Status

The bounded sweep was launched on 2026-09-12 after 48 tests and real-data
training/planning smoke checks passed. Final scientific results are pending.
Smoke-test random images measure throughput only and are never scientific
training data. After `run.py` completes, `finalize.py` measures held-out embedding
spread and isolated regularizer cost, generates the report, and exports a compact
bundle to `experiments/results/two-environment-screen`. These diagnostics never
change coefficient selection. Inspect `finalization.json` for completion/errors.
