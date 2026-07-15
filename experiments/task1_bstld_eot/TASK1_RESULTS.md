# Task 1 — BSTLD EOT Patch and Defense

## Model

Model:
experiments/task1_bstld_eot/models/bstld_sumaiya_best.pt

Classes:
0 = Red
1 = Yellow
2 = Green
3 = False

Model SHA256:
b921a33d0c5b61874f952c6d953f3c1c3f9d0f7d2478cd42e5a6f84dc26b9884

---

## Sumaiya Reference Clean Evaluation

Classification accuracy: 97.41%
Conditional misclassification rate: 2.59%
Vanishing rate: 36.26%

Reference counts:
Total GT red TLs: 182
Detected matched: 116
Correct color: 113
Misclassified: 3
Vanished: 66

NOTE:
These values were extracted from
bstld_clean_data_result.docx.

---

## Local Clean Evaluation

Total GT red TLs: 637
Detected matched: 415
Correct color: 412
Misclassified: 3
Targeted Green: 0
Vanished: 222

Classification accuracy: 99.28%
TMR: 0.00%
Vanishing rate: 34.85%

NOTE:
Local dataset/evaluation population differs from the
182-GT reference population in Sumaiya's document.
Do not silently substitute local clean metrics for the
reference metrics without documenting this difference.

---

## Corrected Smoke EOT Experiment

Training:
Epochs: 1
PGD steps: 2
Learning rate: 0.01
Patch width multiplier: 2.5
EOT: enabled
EOT brightness: 0.6 to 1.4
EOT rotation: -10 to +10 degrees

Evaluation:
Patched GT Red TLs: 635
Detected matched: 531
Correct Red: 127
Targeted Green: 404
Vanished: 104

Classification accuracy: 23.92%
TMR: 76.08%
Vanishing rate: 16.38%

STATUS:
Smoke test only.
NOT a final poster result.

---

## Final BSTLD EOT Attack

Patch:
experiments/task1_bstld_eot/patches/final/bstld_eot_final_15.pt

Training epochs: 15
PGD steps: 20
Learning rate: 0.01
Patch width multiplier: 2.5
Differentiable EOT: enabled

Classification accuracy: PENDING
TMR: PENDING
Vanishing rate: PENDING

Counts:
Patched GT Red TLs: PENDING
Detected matched: PENDING
Correct Red: PENDING
Targeted Green: PENDING
Vanished: PENDING

---

## BSTLD EOT Defense

Defense methodology: PENDING
Defense model: PENDING

Clean classification accuracy: PENDING
Clean TMR: PENDING
Clean vanishing rate: PENDING

Under EOT attack:
Classification accuracy: PENDING
TMR: PENDING
Vanishing rate: PENDING

---

## Task 1 Status

[ ] Final EOT training complete
[ ] Final EOT attack evaluated
[ ] Final patch archived
[ ] Defense dataset prepared
[ ] Defense model trained
[ ] Defense clean evaluation complete
[ ] Defense attack evaluation complete
[ ] BSTLD figures generated
[ ] Results table updated
