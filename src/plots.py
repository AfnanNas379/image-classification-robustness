"""Figures for the report. All are readable in greyscale (markers, line styles and hatching, not only colour).

Owner: Batool (figures).

Usage:
    python -m src.plots --results results --runs runs --out report/figures
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.degradations import CONDITIONS, DEGRADATION_TYPES, load_condition

NAMES = {"plain_cnn": "Plain CNN", "resnet18": "ResNet-18", "vit": "ViT"}
STYLE = {"plain_cnn": dict(color="#1b1b1b", marker="o", linestyle="-"),
         "resnet18": dict(color="#555555", marker="s", linestyle="--"),
         "vit": dict(color="#999999", marker="^", linestyle=":")}
HATCH = {"plain_cnn": "", "resnet18": "///", "vit": "..."}
TYPE_LABEL = {"blur": "Gaussian blur", "noise": "Gaussian noise", "lowres": "Low resolution", "jpeg": "JPEG"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)


def severity_curves(df, value, ylabel, out, name, scale=100):
    """One panel per degradation type: metric vs severity (level 0 = clean), mean +/- std over seeds."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    for ax, t in zip(axes, DEGRADATION_TYPES):
        for m in NAMES:
            sub = df[(df.model == m) & ((df.type == t) | (df.type == "clean"))]
            if sub.empty:
                continue
            g = sub.groupby("level")[value].agg(["mean", "std"]).reset_index()
            ax.errorbar(g.level, scale * g["mean"], yerr=scale * g["std"].fillna(0), capsize=2,
                        markersize=4, linewidth=1.2, label=NAMES[m], **STYLE[m])
        ax.set_title(TYPE_LABEL[t]); ax.set_xticks([0, 1, 2, 3])
    axes[0].set_ylabel(ylabel)
    fig.supxlabel("Severity level (0 = clean)", fontsize=9, y=0.0)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, out, name)


def rd_bars(per_type, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    width = 0.26
    x = np.arange(len(DEGRADATION_TYPES))
    for i, m in enumerate(NAMES):
        g = per_type[per_type.model == m].groupby("type").RD_pct.agg(["mean", "std"]).reindex(DEGRADATION_TYPES)
        ax.bar(x + (i - 1) * width, g["mean"], width, yerr=g["std"].fillna(0), capsize=2, label=NAMES[m],
               color=STYLE[m]["color"], hatch=HATCH[m], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(["Blur", "Noise", "Low res.", "JPEG"])
    ax.set_ylabel("Relative accuracy drop RD (%)"); ax.legend(frameon=False, fontsize=7)
    _save(fig, out, "rd_per_type")


def training_curves(runs, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for m in NAMES:
        logs = [pd.read_csv(p) for p in glob.glob(os.path.join(runs, f"{m}_seed*", "log.csv"))]
        if not logs:
            continue
        acc = np.stack([l.val_acc.to_numpy() for l in logs if len(l) == len(logs[0])]) * 100
        ep = logs[0].epoch.to_numpy()
        ax.plot(ep, acc.mean(0), label=NAMES[m], linewidth=1.2, color=STYLE[m]["color"], linestyle=STYLE[m]["linestyle"])
        ax.fill_between(ep, acc.min(0), acc.max(0), color=STYLE[m]["color"], alpha=0.15)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)"); ax.legend(frameon=False, fontsize=7)
    _save(fig, out, "training_curves")


def degradation_examples(degraded_dir, out, idx=(3, 15, 27)):
    """Grid: rows = example images, columns = the 13 test conditions."""
    names = [c[0] for c in CONDITIONS]
    fig, axes = plt.subplots(len(idx), len(names), figsize=(7.2, 0.62 * len(idx) + 0.3))
    for j, n in enumerate(names):
        x, _ = load_condition(degraded_dir, n)
        for i, k in enumerate(idx):
            ax = axes[i, j]; ax.imshow(x[k], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(n.replace("_", " L"), fontsize=6)
    _save(fig, out, "degradation_examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--degraded", default="data/degraded")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.results, "per_condition.csv"))
    severity_curves(df, "acc", "Accuracy (%)", a.out, "accuracy_vs_severity")
    severity_curves(df, "ece", "ECE (%)", a.out, "ece_vs_severity")
    rd_bars(pd.read_csv(os.path.join(a.results, "per_type.csv")), a.out)
    training_curves(a.runs, a.out)
    if os.path.isdir(a.degraded):
        degradation_examples(a.degraded, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
