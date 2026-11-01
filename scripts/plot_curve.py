"""Plot the Q10 pretraining curve from every log the run has produced.

Sources, concatenated in step order rather than merged:

* ``data/q10_history.csv`` -- steps 1..35,100 (sessions 1-3);
* ``data/kaggle_out/q10/log.csv`` -- the archived session-4 log, if present;
* ``E:/kron_out/session*/log.csv`` -- every session the watcher has downloaded.

Two panels, because the run is read two ways: the training loss says whether the
optimiser kept working, and ``val_bpb_mixed`` is the number that gets quoted.  The
prompt-probe panel of the earlier version is gone with the prompts themselves: the
corpus is prompt-free since step 36,549, so those columns no longer fill.

Corpus switches are marked, not smoothed over: the train loss steps at every
rebuild and the validation curve does not, because the held-out bin never
changes.  Saying which is which in the figure is cheaper than explaining it twice.

Usage::

    python scripts/plot_curve.py
    python scripts/plot_curve.py --out data/curve_q10.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "q10_history.csv"
ARCHIVE = ROOT / "data" / "kaggle_out" / "q10" / "log.csv"
SESSIONS = Path(r"E:/kron_out")
OUT = ROOT / "data" / "curve_q10.png"

EPOCH_STEPS = 172_273          # corpus v5
GPT2_BPB = 1.554
SOUP_BPB = 1.13343
# (step, label) where the corpus changed under the run.  The validation curve is
# unaffected by all of them: the held-out bin is the same bytes since step 1.
SWITCHES = [
    (35_115, "v3: prompts removed", 0.03),
    (36_549, "v4: prose rebalance", 0.11),
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("step") or "").strip()]
    for r in rows:
        r["_src"] = path.parent.parent.name if path.parent.parent.name != "data" else "history"
    return rows


def session_logs() -> list[tuple[str, list[dict]]]:
    out = []
    if SESSIONS.exists():
        for d in sorted(SESSIONS.glob("session*")):
            for cand in d.rglob("log.csv"):
                rows = load(cand)
                if rows:
                    out.append((d.name, rows))
                break
    return out


def column(rows: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        v = (r.get(name) or "").strip()
        if v:
            xs.append(int(r["step"]))
            ys.append(float(v))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if len(y) < k:
        return y
    out = np.empty_like(y)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    hist = load(HISTORY)
    archive = load(ARCHIVE)
    parts = session_logs()
    both = hist + archive + [r for _, rows in parts for r in rows]
    if not both:
        raise SystemExit("no logs found")
    both.sort(key=lambda r: int(r["step"]))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    # Explicit margins: tight_layout left a hand's width of empty canvas above the
    # first panel because the rotated switch labels live outside the axes.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.075,
                        hspace=0.30)

    # ---------------------------------------------------------- 1. train loss
    ax = axes[0]
    x, y = column(both, "loss")
    ax.plot(x, y, color="#b8c4d4", lw=0.7, label="train loss (raw)")
    ax.plot(x, smooth(y, 61), color="#1f4e79", lw=1.9, label="train loss (61-step mean)")
    ax.set_ylabel("cross-entropy, nats/token")
    ax.set_title("Q10 pretraining: 9.94M params, 25 layers, batch 32, fp16, one T4, "
                 "LR 3e-4", fontsize=11, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    # ------------------------------------------------------------ 2. val bpb
    ax = axes[1]
    vx, vy = column(both, "val_bpb_mixed")
    ax.plot(vx, vy, "o-", color="#1f4e79", ms=4, lw=1.6,
            label="val_bpb_mixed (held-out, bare)")
    ex, ey = column(both, "val_bpb_en")
    rx, ry = column(both, "val_bpb_ru")
    ax.plot(ex, ey, "--", color="#c0504d", lw=1.2, label="EN")
    ax.plot(rx, ry, "--", color="#4f81bd", lw=1.2, label="RU")
    if len(vx):
        ax.axhline(GPT2_BPB, color="#7f7f7f", ls=":", lw=1.3)
        ax.annotate(f"GPT-2-124M  {GPT2_BPB:.3f}", (vx[0], GPT2_BPB),
                    textcoords="offset points", xytext=(4, 4), fontsize=9,
                    color="#595959")
        ax.axhline(SOUP_BPB, color="#2e7d32", ls="-.", lw=1.2)
        ax.annotate(f"best+last soup at 35k  {SOUP_BPB:.3f}", (vx[0], SOUP_BPB),
                    textcoords="offset points", xytext=(4, -12), fontsize=9,
                    color="#2e7d32")
    ax.set_ylabel("bits per byte")
    ax.set_xlabel("step")
    ax.set_title("Held-out bpb: blocks inside every (language, domain) bucket, "
                 "bytes counted per batch", fontsize=11, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    # Sessions as shaded bands, so a flat stretch is attributable to a session
    # boundary and not to a bad batch.
    for name, rows in parts:
        s0 = int(rows[0]["step"])
        s1 = int(rows[-1]["step"])
        for a in axes:
            a.axvspan(s0, s1, color="#1f4e79", alpha=0.045)
        axes[0].annotate(name.replace("session", "s"), (s0, 0.06),
                         xycoords=("data", "axes fraction"), fontsize=8,
                         color="#8a8a8a", rotation=90, va="bottom")

    for step, label, height in SWITCHES:
        for a in axes:
            a.axvline(step, color="#c0504d", ls="--", lw=1.0, alpha=0.75)
        axes[0].annotate(label, (step, height), xycoords=("data", "axes fraction"),
                         textcoords="offset points", xytext=(-5, 0), fontsize=8.5,
                         color="#c0504d", ha="right", va="bottom")

    for a in axes:
        a.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    last = int(x[-1]) if len(x) else 0
    fig.suptitle(f"KronLLM Q10 — pretraining to step {last:,} "
                 f"({last / EPOCH_STEPS:.1%} of a {EPOCH_STEPS:,}-step epoch)",
                 fontsize=13, y=0.995)
    fig.savefig(args.out, dpi=150)
    print(f"[out] {args.out}")

    print(f"[train] {len(x)} points, {x[0]}..{x[-1]}, "
          f"mean {y[:200].mean():.3f} -> {y[-200:].mean():.3f}")
    if len(vx):
        print(f"[val]   {len(vx)} mixed points, last {vy[-1]:.4f} at step {vx[-1]:,}")


if __name__ == "__main__":
    main()
