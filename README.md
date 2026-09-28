# Comparative Study of Image Classification Robustness Under Visual Degradation
CS 421 — Comparative Study Track (Track 2)  
King Faisal University, College of Computer Sciences & Information Technology

## Overview
This project investigates which image classification architecture — a plain CNN, 
ResNet-18, or a Vision Transformer (ViT-S/4) — best maintains its accuracy when 
input images are degraded by blur, noise, reduced resolution, or JPEG compression. 
Rather than comparing peak clean-image accuracy alone, the study measures how much 
each model's performance *drops* from its own clean baseline, to isolate genuine 
robustness from differences in raw classification strength.

## Research Question
Which architecture is most robust to real-world image degradation, and does 
robustness depend on the type and severity of degradation applied?

Operationalized into four research questions (RQ1–RQ4) covering accuracy–severity 
trends, relative degradation ranking, architecture × degradation interaction, and 
the architectural mechanisms that best explain the observed differences.

## Team
- Afnan Hashed Ahmed Nasser — 223049806
- Lulu Abdullah Abdulaziz AlMaghlouth — 223036019
- Batool Hassan Jawad Alawadh — 219020376

**Section:** 64
**Supervised by:** Dr. Hala Mohamed Hamdoun

## Phase I — Status: Complete

Phase I established the literature grounding and full experimental design:

- **Literature review** — 19 peer-reviewed sources, thematically organized around 
  architecture families, corruption-robustness benchmarking, the disputed 
  CNN-vs-Transformer robustness evidence, mechanistic explanations (texture bias, 
  frequency sensitivity, positional-embedding scaling, feature collapse), and 
  training-procedure confounds. Concludes with six identified limitations (L1–L6) 
  in existing studies and an explicit research gap.
- **Experimental design** — dataset, models, degradation conditions, metrics, 
  training protocol, six pre-registered hypotheses (H1–H6), and a full analysis 
  plan, specified in detail in `report/`.

### Experimental Design Summary
| Component | Choice |
|---|---|
| Dataset | CIFAR-10 (60,000 32×32 images, 10 classes), native resolution, no pretraining (primary setting) |
| Models | Plain CNN (~1.15M params), ResNet-18 (~11.2M), ViT-S/4 (~12.5M) — parameter-matched |
| Degradations | Gaussian blur, Gaussian noise, reduced resolution, JPEG compression — 3 severities each (13 test conditions) |
| Metrics | Accuracy, absolute/relative drop, CE, rCE, CmCE, ECE, ACD |
| Fairness controls | Shared training recipe across all models; recipe-sensitivity check with each architecture's conventional optimizer |

## Repository Structure
- `report/` — Phase I and Phase II report source (Overleaf/LaTeX, IEEE conference template)
- `src/` — data pipeline, degradation generator, model training/evaluation code (Phase II)
- `results/` — accuracy tables, plots, and comparison outputs (Phase II)

## Status
✅ Phase I: Literature review and experimental design complete.  
🚧 Phase II: Implementation and experiments not yet started.
