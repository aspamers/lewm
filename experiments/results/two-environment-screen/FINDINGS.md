# Findings from the completed 500-update screen

All 24 training runs, validation selection, final tests, and diagnostics completed.
The current evidence does not establish an advantage for either method in real
control. This is a short-budget screen with two model seeds and 16 test goals.

* TwoRoom tuned test success: SIGReg 31.25%, floor + correlation 34.375%,
  random policy 37.5%. The small numerical lead does not establish better learning.
* PushT final success: zero for both methods and random. The combined method's
  single validation success did not carry over to the reserved test goals.
* TwoRoom evaluation embeddings expose a concern: at the selected combined
  weight 0.1, mean coordinate standard deviation is only 0.0054–0.0066,
  versus 0.9026–0.9150 for selected SIGReg. Even combined weight 0.3 yields
  only 0.0496–0.0541. These are near-constant evaluation representations;
  the larger relative entropy rank does not compensate for low absolute spread.
  Training-mode versus evaluation-mode behavior, including BatchNorm running
  statistics, should be diagnosed before attributing this solely to the loss.
* PushT combined embeddings retain appreciable spread (0.5895–0.6089), but
  neither method achieves final goal success under this training/planning budget.
* Isolated regularizer forward/backward GPU medians: SIGReg 1.122 ms,
  combined 1.179 ms. Full training times and peak allocations are similar.
  Thus this implementation has not demonstrated a compute advantage over SIGReg.
  Sub-JEPA has not been timed or trained in this study.

Next scientific priority: diagnose the TwoRoom evaluation spread, establish a
competent baseline with sufficient training and planning, and then compare
learning curves at matched budgets. Longer training alone is not yet a verified
fix. Preserve these test results as screening evidence; subsequent methodological
choices require fresh confirmation rather than repeated tuning on these goals.
