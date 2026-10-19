# Comparative Study of Image Classification Robustness Under Visual Degradation

**CS 421 — Comparative Study Track (Track 2)**
King Faisal University, College of Computer Sciences & Information Technology

We compare how well a **plain CNN** (baseline), **ResNet-18** and a **Vision Transformer (ViT)** keep their accuracy when test images are degraded by **Gaussian blur, Gaussian noise, reduced resolution and JPEG compression**. All three models are trained **from scratch on CIFAR-10 with one shared recipe**, so only the architecture differs.

## Team

**Group 2, Section 64** · Instructor: Dr. Hala Mohamed Hamdoun

| Member | Student ID | Phase II responsibility |
|---|---|---|
| Afnan Hashed Ahmed Nasser | 223049806 | Data pipeline and degraded test sets (`src/data.py`, `src/degradations.py`, `scripts/make_degraded.py`) |
| Lulu Abdullah Abdulaziz AlMaghlouth | 223036019 | Models and training (`src/models.py`, `src/train.py`) |
| Batool Hassan Jawad Alawadh | 219020376 | Evaluation, metrics and figures (`src/evaluate.py`, `src/metrics.py`, `src/plots.py`) |

## Reproducing the results

**Easiest: Google Colab.** Open `notebooks/colab_run.ipynb` in Colab, select a T4 GPU, and run all cells. Checkpoints are saved to Google Drive; if Colab disconnects, run the cells again and training resumes from the last epoch with identical results.

**Locally (GPU recommended):**

```bash
pip install -r requirements.txt
python -m scripts.make_degraded                       # downloads CIFAR-10, builds the 13 test sets
for s in 0 1 2; do for m in plain_cnn resnet18 vit; do
  python -m src.train --model $m --seed $s --out runs
done; done
python -m src.evaluate --runs runs --out results       # metrics + LaTeX tables
python -m src.plots --results results --runs runs --out results/figures
```

**Checks (CPU, no download):** `python -m tests.smoke_test` runs the full pipeline on synthetic images; `python -m tests.resume_check` verifies that an interrupted run resumes to identical weights.

## Experimental setup

| Component | Choice |
|---|---|
| Dataset | CIFAR-10 ([source](https://www.cs.toronto.edu/~kriz/cifar.html)), 32×32, 10 classes; stratified split 45,000 train / 5,000 validation (seed 42) / 10,000 test |
| Preprocessing | Per-channel normalization with training-subset statistics, applied **after** degradation; training augmentation: random crop (4-px zero padding) + horizontal flip |
| Models | Plain CNN (1,148,874 params), ResNet-18 (11,173,962), ViT 4×4 patches, D=384, 7 blocks, 6 heads (12,470,026) |
| Shared recipe | AdamW (β=0.9, 0.999), lr 1e-3, weight decay 0.05, batch 128, 100 epochs, 5-epoch warm-up + cosine decay, gradient clipping 1.0, mixed precision; best clean-validation checkpoint |
| Seeds | 0, 1, 2 for every model (9 runs) |
| Test conditions | Clean + blur σ∈{0.5,1.0,1.5}, noise σ∈{0.04,0.08,0.12}, resolution {24,16,8}px, JPEG Q∈{60,30,10} = 13 sets, generated once (seed 1234) with SHA-256 checksums |
| Metrics | Accuracy, absolute drop ΔA, relative drop RD (primary), CE/mCE vs plain CNN, CmCE (α=0.1), ECE (15 bins), ACD |

All settings are in [`configs/default.yaml`](configs/default.yaml).

## Repository structure

```
configs/default.yaml      shared configuration for all models
src/                      data, degradations, models, training, evaluation, metrics, plots
scripts/make_degraded.py  generates the 13 fixed test sets
notebooks/colab_run.ipynb end-to-end Colab notebook
tests/                    smoke test and resume check
runs/                     per-run training logs and metadata (checkpoints not committed)
results/                  metric CSVs, LaTeX tables and figures
report/                   Phase I and Phase II reports (main.tex, references.bib, figures/)
```

## Status

- ✅ Phase I: literature review and experimental design.
- 🚧 Phase II: code complete and tested; training and evaluation in progress.
