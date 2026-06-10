# Project Pipeline

## Overall Goal

The project studies whether adversarial patches can fool traffic light recognition models in autonomous vehicle systems, and whether adversarial training can improve robustness.

---

## Step 1: Train Clean YOLOv8 Model

Script/Command:
YOLO training command

Purpose:
Train a normal traffic light detector on clean BSTLD data.

Output:
Clean baseline model weights.

---

## Step 2: Evaluate Clean Accuracy

Script:
evaluate-clean-acc.py

Purpose:
Measure how well the clean model performs before any attack.

Metrics:
- Accuracy
- Misclassification rate
- Vanishing rate

---

## Step 3: Train Adversarial Patch

Script:
train_patch_yolov8.py

Purpose:
Create a patch that causes the model to misclassify traffic lights.

Example:
Red → Green

Output:
Trained patch file.

---

## Step 4: Evaluate Patch Attack

Script:
evaluate_patch_eot.py

Purpose:
Test whether the adversarial patch works under transformations like brightness, rotation, and resizing.

Metrics:
- Successful attacks
- Failed attacks
- Vanishing detections

---

## Step 5: Adversarial Defense Training

Script:
adversarial-defense.py

Purpose:
Train the model using patched examples so it becomes more robust.

Output:
Robust/defended model weights.

---

## Step 6: Re-Evaluate Robust Model

Scripts:
evaluate-clean-acc.py
evaluate_patch_eot.py

Purpose:
Check whether the defended model improves performance against the patch.

---

## My Understanding

The full workflow is:

Clean Model
↓
Clean Evaluation
↓
Patch Training
↓
Patch Evaluation
↓
Defense Training
↓
Robust Model Evaluation