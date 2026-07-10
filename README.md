# Adversarial Patch Attacks and Defense on Traffic Light Detection

REU Research Project — The University of Alabama

## Overview

This project investigates the vulnerability of YOLOv8m-based traffic light detectors to EOT adversarial patch attacks, and evaluates whether patch-augmented adversarial training can restore robustness.

**Key Results (LISA Dataset):**
| Metric | Baseline | Defense |
|---|---|---|
| Attack Success (TMR) | 72.9% | 19.6% |
| Total Affected | 76.1% | 23.1% |
| Clean mAP50 | 0.958 | 0.965 |

## Project Structure

- `scripts/` — training, evaluation, and figure generation scripts
- `configs/` — YAML config files for training and attack
- `patch_utils/` — patch transformation and loss utilities
- `final_results/paper_figures/` — all paper-ready figures
- `classnames/` — class name files for each dataset

## Usage

### Train the adversarial patch
```bash
python scripts/train_patch_yolov8_fixed.py configs/lisa_remapped_attack_fast.yaml
```

### Evaluate patch against baseline and defense models
```bash
python scripts/evaluate_final_patch.py
```

### Retrain BSTLD model
```bash
python scripts/retrain_bstld.py
```

### Generate paper figures
```bash
python scripts/generate_final_figures.py
```

## Models

- LISA Baseline: `runs/detect/lisa_remapped_yolov8m/weights/best.pt`
- LISA Defense: `runs/detect/lisa_defense_yolov8m_50/weights/best.pt`
- BSTLD: `runs/detect/bstld_yolov8m_final3/weights/best.pt`

## Based On

Pavlitska et al., "Fool the Stoplight: Realistic Adversarial Patch Attacks on Traffic Light Detectors," arXiv:2506.04823, 2025.
