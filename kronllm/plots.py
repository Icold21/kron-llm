"""Plotting helpers shared by the dashboard and notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"train": "#3b82f6", "val": "#ef4444", "diag": "#10b981"}


def read_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df


def _ensure_axes(fig, n: int = 1):
    """Reuse the axes of a long-lived figure (avoids leaking figures in the UI)."""
    while len(fig.axes) < n:
        fig.add_subplot(1, n, len(fig.axes) + 1)
    return fig.axes


def loss_curves(log_path: str | Path, smooth: int = 10, fig=None):
    """Train (smoothed) + validation loss from a training ``log.csv``."""
    df = read_log(log_path)
    if df.empty or "split" not in df:
        return None
    if fig is None:
        fig = plt.figure(figsize=(7, 3.6))
    ax = _ensure_axes(fig, 1)[0]
    ax.clear()
    tr = df[df["split"] == "train"].dropna(subset=["loss"])
    va = df[df["split"] == "val"].dropna(subset=["loss"])
    if len(tr):
        y = tr["loss"].rolling(min(smooth, max(1, len(tr))), min_periods=1).mean()
        ax.plot(tr["step"], tr["loss"], color=COLORS["train"], alpha=0.25, lw=1)
        ax.plot(tr["step"], y, color=COLORS["train"], label="train (smooth)")
    if len(va):
        ax.plot(va["step"], va["loss"], color=COLORS["val"], marker="o",
                ms=3, lw=1.5, label="val")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy")
    ax.set_title("loss curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def diag_curves(log_path: str | Path, fig=None):
    df = read_log(log_path)
    if df.empty or "split" not in df:
        return None
    dg = df[df["split"] == "diag"]
    if dg.empty:
        return None
    cols = [c for c in ("sigma_theta_mean", "sigma_theta_var", "attn_entropy_mean",
                        "h_norm_ratio_mean", "eff_rank_mean", "eff_rank_mean",
                        "q_drift_mean") if c in dg]
    if not cols:
        return None
    if fig is None:
        fig = plt.figure(figsize=(7, 3.2))
    ax = _ensure_axes(fig, 1)[0]
    ax.clear()
    for c in cols:
        ax.plot(dg["step"], dg[c], marker=".", label=c)
    ax.set_xlabel("step")
    ax.set_title("diagnostics")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def read_summary(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def compare_plot(results_dir: str | Path, fig=None):
    df = read_summary(results_dir)
    if df.empty:
        return None
    df = df.dropna(subset=["val_ppl"])
    if fig is None:
        fig = plt.figure(figsize=(11, 3.8))
    axes = _ensure_axes(fig, 2)
    for a in axes[:2]:
        a.clear()
    exp = df["experiment"].astype(str)
    axes[0].bar(exp, df["val_ppl"], color="#6366f1")
    axes[0].set_ylabel("val perplexity")
    axes[0].set_title("E-series: validation PPL")
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].scatter(df["n_params"], df["val_ppl"], c="#ef4444", s=40)
    for _, row in df.iterrows():
        axes[1].annotate(str(row["experiment"]),
                         (row["n_params"], row["val_ppl"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("parameters")
    axes[1].set_ylabel("val perplexity")
    axes[1].set_title("quality vs. budget")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig


def results_table(results_dir: str | Path) -> pd.DataFrame:
    df = read_summary(results_dir)
    if df.empty:
        return pd.DataFrame({"info": ["no experiments finished yet"]})
    cols = [c for c in ("experiment", "n_params", "n_params_matrix", "val_ppl",
                        "ppl_ratio_vs_E0", "sec_per_step", "eff_rank_D",
                        "eff_rank_W", "eff_rank_P", "cos_prev_max",
                        "tau_mean", "gate_mean", "eff_rank_spearman",
                        "sigma_theta_var", "peak_vram_gb") if c in df]
    return df[cols].sort_values("val_ppl")
