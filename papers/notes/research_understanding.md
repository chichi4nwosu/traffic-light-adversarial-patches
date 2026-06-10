# Research Understanding Check

## What is the research problem?

Can adversarial patches fool traffic light detectors used in autonomous vehicles?

---

## What dataset is used?

BSTLD

---

## What model is attacked?

YOLOv8

---

## What attack is used?

Adversarial Patch + EOT

---

## What is EOT?

Expectation Over Transformation

Uses:
- brightness
- rotation
- padding
- resizing

---

## What is the goal of the attack?

Cause traffic light misclassification.

Examples:
- Red → Green
- Red → Off
- Traffic light vanishes

---

## What defense is used?

Adversarial training

---

## What metrics are used?

- Accuracy
- Misclassification Rate
- Vanishing Rate

---

## What are the outputs?

- Trained patch
- Attack evaluation results
- Robust model
- Graphs