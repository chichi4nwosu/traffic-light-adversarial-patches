# EOT Evaluation Notes

## What is the input?
- config file
- trained adversarial patch
- YOLO model

## What does this script test?
It tests whether the adversarial patch can still fool the model after random transformations.

## What transformations are applied?
- brightness changes
- padding
- rotation
- resizing

## What counts as a successful attack?
A successful attack happens when the patched traffic light is predicted as the target class.

Example:
Red light → Green light

## What counts as failure?
The model still predicts the original class correctly.

## What is vanishing?
The traffic light disappears from detection completely.

## Why this matters
This is the script that gives the main attack results for the poster/paper.

# EOT Evaluation Notes

## What is this script for?

This script evaluates the trained adversarial patch under EOT conditions.

EOT means Expectation Over Transformation. The patch is tested with random changes like brightness, padding, rotation, and resizing to see if it still works under realistic conditions.

---

## What are the inputs?

- Config file
- Trained patch file
- YOLO model
- Validation dataset

---

## What transformations are used?

- Brightness change
- Padding
- Rotation
- Resizing

These simulate real-world changes such as lighting, distance, and camera angle.

---

## What counts as a successful attack?

A successful attack happens when the original traffic light class is changed into the target class.

Example:

Red light → Green light

---

## What counts as failed?

The attack fails if the model still predicts the original class correctly.

Example:

Red light → Red light

---

## What is vanishing?

Vanishing happens when the model does not detect the traffic light at all after the patch is applied.

---

## What outputs does the script save?

The script saves examples of:

- Successful attacks
- Failed attacks
- Clean images
- Attacked images

---

## Why this matters

This script produces the main evidence for whether the adversarial patch is effective. The results from this script will likely become part of the poster and paper.

# Adversarial Defense Notes

## What is this script for?

This script trains a more robust YOLO model using adversarially patched images.

The goal is to teach the model not to be fooled by the adversarial patch.

---

## What are the inputs?

- Config file
- Trained adversarial patch
- YOLO model weights
- Training dataset

---

## What is the defense strategy?

The defense uses adversarial training.

This means the model is trained on a mixture of:
- clean images
- patched/adversarial images

---

## What is being trained?

Unlike `train_patch_yolov8.py`, this script trains the model weights.

The patch is already trained and is kept fixed/frozen.

---

## What does “frozen patch” mean?

The patch is not changing anymore.

Instead, the model is learning how to correctly classify traffic lights even when that patch is present.

---

## Why does this matter?

If the model sees adversarial examples during training, it may become better at resisting similar attacks later.

---

## What is the output?

The script saves a defended/robust model.

Example output:

outputs-defense-robust/bstld_robust.pt

---

## My understanding

First, the project creates an adversarial patch to fool YOLO.

Then, this script uses that patch to train YOLO to become more resistant.

So the defense pipeline is:

Clean model
↓
Adversarial patch
↓
Patched training images
↓
Robust model
↓
Re-evaluate attack