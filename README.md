# Adversarial Patch Attacks and Defense on Traffic Light Detection

REU Research Project — The University of Alabama

## Overview

This project investigates the vulnerability of YOLOv8m-based traffic light detectors to Expectation-over-Transformation (EOT) adversarial patch attacks, and evaluates whether patch-augmented adversarial training can restore robustness. Built on top of [KASTEL-MobilityLab/attacks-on-traffic-light-detection](https://github.com/KASTEL-MobilityLab/attacks-on-traffic-light-detection), extended with BSTLD EOT attacks, adaptive attacks, defense training, and transferability evaluation.

## Key Results (LISA Dataset)

| Metric | Baseline | Defense |
|---|---|---|
| Attack Success (TMR) | 72.9% | 19.6% |
| Total Affected | 76.1% | 23.1% |

Patch-augmented adversarial training reduces attack success rate by ~53 percentage points while preserving clean-data detection performance.

## Repository Structure
