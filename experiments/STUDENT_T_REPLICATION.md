# Student-t training-seed replication

Protocol fixed before running: four targets (Gaussian and Student-t df 3, 5, 9),
weight 0.09 for every target, 1,000 updates, ten new training seeds 3 through 12.
Keep the original architecture, optimizer, minibatches per seed, training data,
validation data, test data seed 48000 and 24 planning goals unchanged. No target
or coefficient is selected from this replication. This is 40 new training runs.

Primary outcome: mean final planning goal distance (lower is better).
Primary contrast: df5 distance minus the average of df3 and df9 distances,
paired within training seed. Positive means the previously observed df5 dip
persists. Report a deterministic 20,000-resample paired seed bootstrap 95%
percentile interval. This interval measures training-seed uncertainty conditional
on these fixed data and goals; it is not uncertainty over environments or tasks.
Ten seeds remain a small sample. The contrast was motivated by the original
three-seed results and is assessed on new training seeds, not a fresh dataset.

Secondary outcomes: planning success, each Student-t minus Gaussian distance,
learned variance and tail diagnostics. Secondary intervals are descriptive,
without multiple-comparison correction. Report the original three seeds and
all 13 pooled seeds separately from the primary ten-seed replication.
Do not treat repeated evaluation of the same goals as independent tasks.

Run with the repository src directory on PYTHONPATH:

```powershell
python experiments/student_t_replication.py
python experiments/report_student_t_replication.py outputs/student-t-replication-1000
```

This is a small synthetic pilot, not the published LeWM/TwoRoom benchmark.
