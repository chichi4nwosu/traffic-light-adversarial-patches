# Adversarial Patch Attacks, Defenses, and Transferability for Traffic Light Detection

> University of Alabama NSF REU Research Project

<p align="center">
  <img src="publication/figures/system_design/system_pipeline.png" width="900">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Research](https://img.shields.io/badge/Research-Adversarial%20Machine%20Learning-orange)
![License](https://img.shields.io/badge/License-AGPL--3.0-lightgrey)

</p>

---

## Overview

This repository contains the implementation of **physical adversarial patch attacks**, **adversarial defenses**, **adaptive attacks**, and **cross-model transferability experiments** for YOLOv8 traffic-light detection.

The project investigates the robustness of traffic-light perception systems used in autonomous vehicles against physically realizable adversarial patches.

The complete pipeline includes

- Physical adversarial patch optimization
- Expectation over Transformation (EOT)
- Patch-based adversarial training
- Adaptive attack generation
- Cross-model transferability analysis
- Evaluation on multiple datasets
- Publication-quality figures and experiments

---

# Highlights

✔ Physical adversarial patch attacks

✔ Patch-based defense training

✔ Adaptive attack evaluation

✔ Cross-model transferability

✔ YOLOv8m and YOLOv8l evaluation

✔ LISA and BSTLD datasets

✔ Fully reproducible experiments

---

# Datasets

## LISA Traffic Light Dataset

The LISA dataset contains annotated traffic-light images collected from real driving scenes in the United States.

Classes

- Red
- Yellow
- Green
- Background

Used for

- Baseline attack
- Defense evaluation
- Adaptive attacks

---

## Bosch Small Traffic Lights Dataset (BSTLD)

The Bosch Small Traffic Lights Dataset (BSTLD) contains high-resolution urban driving scenes with small traffic lights.

Characteristics

- High-resolution images
- Small object detection
- Challenging lighting conditions
- Real-world traffic scenes

Used for

- Attack generation
- Defense evaluation
- Cross-model transferability

---

# Repository Structure

```
traffic-light-adversarial-patches/

│
├── configs/
│
├── scripts/
│
├── patch_utils/
│
├── datasets/
│
├── experiments/
│   │
│   ├── task1_bstld_eot/
│   │
│   ├── task2_bstld_transferability/
│   │
│   ├── defense/
│   │
│   └── archive/
│
├── publication/
│   │
│   ├── paper/
│   │
│   ├── poster/
│   │
│   ├── figures/
│   │
│   └── tables/
│
├── results/
│
├── runs/
│
├── README.md
│
└── requirements.txt
```

---

# System Design

The overall research pipeline is shown below.

```
Dataset

↓

YOLOv8 Detector

↓

Adversarial Patch Optimization

↓

Physical Patch

↓

Evaluation

↓

Defense Training

↓

Adaptive Attack

↓

Transferability Evaluation

↓

Final Results
```

---

# Experimental Pipeline

## Task 1

Baseline Adversarial Patch Attack

- Train YOLOv8 detector
- Generate adversarial patch
- Evaluate attack
- Evaluate defense
- Adaptive attack evaluation

---

## Task 2

Cross-Model Transferability

Train adversarial patch on

YOLOv8m

↓

Evaluate on

YOLOv8l

Then

Train patch on

YOLOv8l

↓

Evaluate on

YOLOv8m

---

# Results

## BSTLD

| Experiment | Classification Accuracy | Targeted Misclassification Rate | Vanishing Rate |
|------------|-----------------------:|-------------------------------:|---------------:|
| Original Attack | 22.98% | 77.02% | 16.38% |
| Defended Model (Fixed Attack) | **100.00%** | **0.00%** | 13.07% |
| Defended Model (Adaptive Attack) | 71.74% | 28.26% | 27.56% |

---

## YOLOv8m Source Attack

| Epoch | Accuracy | TMR | Vanishing |
|------:|---------:|----:|----------:|
| 1 | 32.23 | 67.58 | 19.37 |
| 2 | 31.20 | 68.80 | 18.74 |
| 3 | 30.83 | 68.97 | 20.31 |
| 4 | 30.52 | 69.29 | 17.95 |
| **5** | **29.24** | **70.76** | **19.21** |

---

## Transferability (YOLOv8m → YOLOv8l)

| Model | Accuracy | Targeted Misclassification | Vanishing |
|-------|---------:|---------------------------:|-----------:|
| YOLOv8l | **18.16%** | **81.42%** | 24.57% |

---

## YOLOv8l Source Attack

| Epoch | Accuracy | TMR | Vanishing |
|------:|---------:|----:|----------:|
| 1 | 44.73 | 54.27 | 20.79 |
| 2 | 44.80 | 54.40 | 21.26 |
| 3 | 42.88 | 55.95 | 19.21 |
| **4** | **40.99** | **58.02** | **20.47** |
| 5 | 41.11 | 57.91 | 20.31 |

---

# Qualitative Examples

## Original Detection

<img src="publication/figures/qualitative/original.png" width="700">

---

## Adversarial Patch

<img src="publication/figures/qualitative/patched.png" width="700">

---

## Defense

<img src="publication/figures/qualitative/defended.png" width="700">

---

# Installation

```bash
git clone https://github.com/<username>/traffic-light-adversarial-patches.git

cd traffic-light-adversarial-patches

python -m venv traffic-env

source traffic-env/bin/activate

pip install -r requirements.txt
```

---

# Reproducing Experiments

## Train YOLO Detector

```bash
python train_detector.py
```

---

## Generate Adversarial Patch

```bash
python experiments/task1_bstld_eot/scripts/train_bstld_eot.py \
experiments/task1_bstld_eot/configs/bstld_eot_corrected_final.yaml
```

---

## Evaluate Patch

```bash
python experiments/task1_bstld_eot/evaluation/evaluate_bstld_final_patch.py
```

---

## Evaluate Defense

```bash
python experiments/task1_bstld_eot/defense/evaluate_bstld_defended_patch.py
```

---

## Evaluate Transferability

```bash
python experiments/task2_bstld_transferability/evaluation/evaluate_task2_yolov8l_transfer.py
```

---

# Future Work

- Cross-dataset transferability
- Physical-world printed patch evaluation
- Multi-object attacks
- Universal adversarial patches
- Transformer-based traffic-light detectors
- Real-time defense mechanisms

---

# Citation

If you use this repository, please cite our forthcoming publication.

```bibtex
@misc{nwosu2026trafficlight,
  title={Adversarial Patch Attacks, Defenses, and Transferability for Traffic Light Detection},
  author={Chinualumogu Nwosu and Sumaiya...},
  year={2026},
  note={NSF REU Research Project}
}
```

---

# Acknowledgements

This work was completed as part of the NSF Research Experiences for Undergraduates (REU) program at the University of Alabama.

Advisor

- Sumaiya Tasneem

Built using

- PyTorch
- Ultralytics YOLOv8
- Kornia
- OpenCV

---

# References

1. Pavlitskaya et al., *Adversarial Patch Attacks on Object Detection Systems*

2. Bosch Small Traffic Lights Dataset

3. LISA Traffic Light Dataset

4. Ultralytics YOLOv8 Documentation

5. PyTorch Documentation

6. Kornia Documentation

---

## License

This repository is released under the AGPL-3.0 License.
