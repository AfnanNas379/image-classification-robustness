# Comparative Study of Image Classification Robustness Under Visual Degradation

**CS 421 — Comparative Study Track (Track 2)**
King Faisal University, College of Computer Sciences & Information Technology

## Overview

This project investigates which image classification architecture — a plain CNN, ResNet-18, or a Vision Transformer (ViT) — best maintains its accuracy when input images are degraded by blur, noise, reduced resolution, or JPEG compression. Rather than comparing clean-image accuracy alone, the study measures how much each model's performance drops from its own clean baseline, to separate genuine robustness from differences in raw classification strength.

## Research Question

Which architecture is most robust to real-world image degradation, and does robustness depend on the type and severity of degradation applied?

This is broken down into four research questions:

- **RQ1:** How does accuracy change with severity for each degradation?
- **RQ2:** Which architecture shows the smallest relative drop after controlling for clean accuracy?
- **RQ3:** Does the robustness ranking depend on the degradation type?
- **RQ4:** Which architectural properties best explain the differences?

## Team

**Group [NN], Section 64**

| Name | Student ID |
|---|---|
| Afnan Hashed Ahmed Nasser | 223049806 |
| Lulu Abdullah Abdulaziz AlMaghlouth | 223036019 |
| Batool Hassan Jawad Alawadh | 219020376 |

**Instructor:** Dr. Hala Mohamed Hamdoun

## Phase I — Literature Survey and Experimental Design

- **Literature review:** 10 peer-reviewed studies, organized into three themes: robustness benchmarking, CNN-versus-Transformer robustness, and mechanisms and training factors (frequency sensitivity, patch/positional-embedding effects, pretraining and training-recipe confounds). The review concludes with five limitations (L1–L5) of existing experimental studies and explains why a new, controlled comparison is needed.
- **Experimental design:** task definition, dataset, candidate models and baseline, degradation conditions, and evaluation metrics.

## Experimental Design Summary

| Component | Choice |
|---|---|
| Dataset | CIFAR-10 (60,000 images, 32×32, 10 classes) at native resolution; 45,000 train / 5,000 validation / 10,000 test |
| Models | Plain CNN (~1.15M params, baseline), ResNet-18 (~11.2M), ViT with 4×4 patches (~12.5M); ResNet-18 and ViT are parameter-matched, the plain CNN is a compact baseline |
| Training | All models trained from scratch (no pretraining) under one shared training recipe |
| Degradations | Gaussian blur, Gaussian noise, reduced resolution, JPEG compression — 3 severity levels each, plus clean (13 test conditions) |
| Metrics | Accuracy, absolute and relative accuracy drop, CE, CmCE, ECE, ACD |

## Repository Structure

```
report/    Phase I report (PDF and LaTeX source)
src/       Data pipeline, degradation generator, training and evaluation code (planned for Phase II)
results/   Accuracy tables, plots, and comparison outputs (planned for Phase II)
```

## Status

- ✅ **Phase I:** Literature review and experimental design complete.
- 🚧 **Phase II:** Implementation and experiments not yet started.
