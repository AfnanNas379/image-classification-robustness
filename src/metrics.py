"""Robustness and confidence metrics (Section III-E of the Phase I report).

Owner: Batool (metrics).

Notation: Acc in [0, 1]; error E = 1 - Acc. Reported in % where stated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width confidence bins: sum_b |B|/N * |acc(B) - conf(B)|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def add_drops(per_cond: pd.DataFrame) -> pd.DataFrame:
    """Add absolute drop dA (percentage points) and relative drop RD (%) to every row.

    per_cond columns: model, seed, condition, type, level, acc, mean_conf, ece
    """
    df = per_cond.copy()
    clean = df[df.type == "clean"].set_index(["model", "seed"]).acc
    base = df.set_index(["model", "seed"]).index.map(clean)
    df["clean_acc"] = np.asarray(base)
    df["dA_pp"] = 100 * (df.clean_acc - df.acc)                    # Eq. (2)
    df["RD_pct"] = 100 * (df.clean_acc - df.acc) / df.clean_acc    # Eq. (3)
    return df


def corruption_errors(df: pd.DataFrame, reference: str = "plain_cnn") -> pd.DataFrame:
    """CE and relative CE per degradation type, against the reference model (same seed).

    CE_c^m    = sum_s E^m_{c,s} / sum_s E^ref_{c,s}
    rCE_c^m   = sum_s (E^m_{c,s} - E^m_clean) / sum_s (E^ref_{c,s} - E^ref_clean)
    """
    deg = df[df.type != "clean"].copy()
    deg["err"] = 1 - deg.acc
    deg["err_gap"] = deg.clean_acc - deg.acc
    g = deg.groupby(["model", "seed", "type"])[["err", "err_gap"]].sum().reset_index()
    ref = g[g.model == reference].set_index(["seed", "type"])
    g["CE"] = g.err / g.set_index(["seed", "type"]).index.map(ref.err).values
    g["rCE"] = g.err_gap / g.set_index(["seed", "type"]).index.map(ref.err_gap).values
    return g[["model", "seed", "type", "CE", "rCE"]]


def cmce(df: pd.DataFrame, alpha: float = 0.10) -> pd.DataFrame:
    """Calibrated mean Corruption Error (Liu et al., 2024): (1 - mE - alpha) / (Acc_clean - alpha).

    mE = mean error over all 12 degraded conditions. 1 = no degradation, 0 = chance level.
    """
    deg = df[df.type != "clean"]
    out = deg.groupby(["model", "seed"]).agg(mE=("acc", lambda a: 1 - a.mean()),
                                            clean_acc=("clean_acc", "first")).reset_index()
    out["CmCE"] = (1 - out.mE - alpha) / (out.clean_acc - alpha)
    return out


def acd(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy-Confidence Divergence (Erkara & Mayer, 2026), per degradation type.

    D_c = mean_s (Acc_{c,s} - Conf_{c,s})      (negative = overconfident)
    N_c = #levels whose gap has the same sign as D_c
    """
    rows = []
    for (m, s, c), g in df[df.type != "clean"].groupby(["model", "seed", "type"]):
        gap = (g.acc - g.mean_conf).to_numpy()
        d = gap.mean()
        rows.append({"model": m, "seed": s, "type": c, "D_c": 100 * d,
                     "N_c": int((np.sign(gap) == np.sign(d)).sum())})
    return pd.DataFrame(rows)


def summarize(per_cond: pd.DataFrame, reference: str = "plain_cnn", alpha: float = 0.10):
    """Return a dict of tidy tables (per seed) used for the report."""
    df = add_drops(per_cond)
    per_type = (df[df.type != "clean"].groupby(["model", "seed", "type"])[["dA_pp", "RD_pct", "ece"]]
                .mean().reset_index())
    overall = (df[df.type != "clean"].groupby(["model", "seed"])[["dA_pp", "RD_pct", "ece"]]
               .mean().reset_index())
    clean = df[df.type == "clean"][["model", "seed", "acc", "ece"]].rename(
        columns={"acc": "clean_acc", "ece": "clean_ece"})
    ce = corruption_errors(df, reference)
    mce = ce.groupby(["model", "seed"])[["CE", "rCE"]].mean().reset_index().rename(
        columns={"CE": "mCE", "rCE": "rmCE"})
    overall = overall.merge(clean, on=["model", "seed"]).merge(mce, on=["model", "seed"]) \
                     .merge(cmce(df, alpha)[["model", "seed", "CmCE"]], on=["model", "seed"])
    return {"per_condition": df, "per_type": per_type, "overall": overall, "ce": ce, "acd": acd(df)}


def mean_std(table: pd.DataFrame, by, cols) -> pd.DataFrame:
    """Aggregate over seeds: mean and standard deviation of each column."""
    agg = table.groupby(by)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()
