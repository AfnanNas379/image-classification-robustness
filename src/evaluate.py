"""Evaluate every trained model on all 13 test conditions and write the result tables.

Owner: Batool (evaluation).

Usage:
    python -m src.evaluate --runs runs --out results
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from src import data as D
from src.degradations import CONDITIONS, load_condition
from src.metrics import expected_calibration_error, mean_std, summarize
from src.models import build_model


@torch.no_grad()
def predict(model, norm, x_uint8, batch_size, dev):
    model.eval()
    probs = []
    for i in range(0, len(x_uint8), batch_size):
        xb = D.to_tensor(x_uint8[i:i + batch_size]).to(dev)
        probs.append(torch.softmax(model(norm(xb)).float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def evaluate_run(run_dir, cfg, dev):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=dev, weights_only=False)
    model = build_model(meta["model"], cfg["models"].get(meta["model"], {})).to(dev)
    model.load_state_dict(ck["model"])
    norm = D.Normalizer(ck["mean"], ck["std"]).to(dev)
    rows, preds = [], {}
    for name, kind, level, _ in CONDITIONS:
        x, y = load_condition(cfg["data"]["degraded_dir"], name)
        p = predict(model, norm, x, cfg["eval"]["batch_size"], dev)
        pred, conf = p.argmax(1), p.max(1)
        correct = (pred == y)
        rows.append({"model": meta["model"], "seed": meta["seed"], "condition": name, "type": kind,
                     "level": level, "acc": correct.mean(), "mean_conf": conf.mean(),
                     "ece": expected_calibration_error(conf, correct, cfg["eval"]["ece_bins"]),
                     "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())})
        preds[name] = np.stack([pred, (conf * 1e4).round()], 1).astype(np.int32)
        print(f"  {name:9s} acc={rows[-1]['acc']:.4f} conf={rows[-1]['mean_conf']:.3f} ece={rows[-1]['ece']:.4f}")
    return rows, preds


def latex_table(df: pd.DataFrame, cols, headers, caption, label, fmt="{:.2f}") -> str:
    """booktabs table with mean ± std (the report requires booktabs and units on every column)."""
    lines = [r"\begin{table}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(headers) + r" \\", r"\midrule"]
    names = {"plain_cnn": "Plain CNN", "resnet18": "ResNet-18", "vit": "ViT"}
    for _, r in df.iterrows():
        cells = [f"{fmt.format(r[c + '_mean'])} $\\pm$ {fmt.format(r[c + '_std'])}" for c in cols]
        lines.append(names.get(r.model, r.model) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(a.out, "predictions"), exist_ok=True)

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(a.runs, "*_seed*"))):
        if not os.path.exists(os.path.join(run_dir, "done.json")):
            print(f"[skip unfinished] {run_dir}"); continue
        print(f"[eval] {run_dir}")
        r, preds = evaluate_run(run_dir, cfg, dev)
        rows += r
        np.savez_compressed(os.path.join(a.out, "predictions", os.path.basename(run_dir) + ".npz"), **preds)

    per_cond = pd.DataFrame(rows)
    per_cond.to_csv(os.path.join(a.out, "per_condition.csv"), index=False)
    S = summarize(per_cond, cfg["eval"]["reference_model"], cfg["eval"]["chance_accuracy"])
    for k, t in S.items():
        t.to_csv(os.path.join(a.out, f"{k}.csv"), index=False)

    order = {"plain_cnn": 0, "resnet18": 1, "vit": 2}
    ov = S["overall"].copy()
    for c in ("clean_acc", "clean_ece", "ece"):
        ov[c] = 100 * ov[c]
    ov_ms = mean_std(ov, "model", ["clean_acc", "dA_pp", "RD_pct", "mCE", "CmCE", "ece"])
    ov_ms = ov_ms.sort_values("model", key=lambda s: s.map(order))
    ov_ms.to_csv(os.path.join(a.out, "overall_mean_std.csv"), index=False)
    pt = S["per_type"].pivot_table(index=["model", "seed"], columns="type", values="RD_pct").reset_index()
    pt_ms = mean_std(pt, "model", ["blur", "noise", "lowres", "jpeg"]).sort_values(
        "model", key=lambda s: s.map(order))
    pt_ms.to_csv(os.path.join(a.out, "rd_per_type_mean_std.csv"), index=False)

    with open(os.path.join(a.out, "tables.tex"), "w") as f:
        f.write(latex_table(ov_ms, ["clean_acc", "dA_pp", "RD_pct", "mCE", "CmCE", "ece"],
                            ["Clean Acc. (\\%)", "$\\Delta A$ (pp)", "RD (\\%)", "mCE (ratio)",
                             "CmCE (0--1)", "ECE (\\%)"],
                            "Overall results (mean $\\pm$ std over 3 seeds; degraded metrics averaged over 12 conditions)",
                            "tab:overall") + "\n\n")
        f.write(latex_table(pt_ms, ["blur", "noise", "lowres", "jpeg"],
                            ["Blur (\\%)", "Noise (\\%)", "Low res. (\\%)", "JPEG (\\%)"],
                            "Relative accuracy drop RD per degradation type (\\%, mean $\\pm$ std over 3 seeds; lower is more robust)",
                            "tab:rd_type"))
    print(ov_ms.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
