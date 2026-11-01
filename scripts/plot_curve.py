"""Plot the Q10 pretraining curve from the two logs that hold it.

`data/q10_history.csv` covers steps 1-35,100 (sessions 1-3, collected from the
per-session logs as they came back); `data/kaggle_out/q10/log.csv` covers
35,150-43,895 (session 4).  They are concatenated here rather than merged into
one file, because the session-4 log also carries the prompt probe columns that
the earlier sessions never wrote.

Three panels, because the run is read three different ways:

* training loss -- did the optimiser keep working, and where did the step time
  change;
* `val_bpb_mixed` -- the metric that is actually quoted, with the per-language
  split and the GPT-2-124M reference on the same blocks;
* the prompt probe -- the evidence that the step-35,115 discontinuity is the
  model learning the `<|system|>` tag, not a regression.

Usage::

    python scripts/plot_curve.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HISTORY = Path("data/q10_history.csv")
SESSION = Path("data/kaggle_out/q10/log.csv")
OUT = Path("data/curve_q10.png")

# Step at which the fully-prompted corpus and the fixed resume both took effect.
CORPUS_SWITCH = 35_115
# Averaged checkpoint: best.pt + last.pt, alpha 0.5, 1024 strided blocks.
SOUP_BPB = 1.13343
GPT2_BPB = 1.554


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)]


def column(rows: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        v = (r.get(name) or "").strip()
        if v:
            xs.append(int(r["step"]))
            ys.append(float(v))
    return np.array(xs), np.array(ys)


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Rolling mean that keeps the ends honest (shrinking window, not NaN)."""
    if len(y) < k:
        return y
    out = np.empty_like(y)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


def main() -> None:
    hist = load(HISTORY)
    sess = load(SESSION)
    both = hist + sess

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.0, 0.9]})

    # ---------------------------------------------------------- 1. train loss
    ax = axes[0]
    x, y = column(both, "loss")
    ax.plot(x, y, color="#b8c4d4", lw=0.8, label="train loss (raw)")
    ax.plot(x, smooth(y, 41), color="#1f4e79", lw=1.8,
            label="train loss (41-step mean)")

    # The prompt tokens are inside the training loss, so that loss is flattered
    # by ~1 % of the stream becoming cheap for free.  Correcting it with the
    # probe is what makes the two sessions comparable; the corrected line sits
    # almost on top of the raw one, which is the point.
    px_, py_ = column(sess, "nats_prompt")
    sx_, sy_ = column(sess, "loss")
    if len(px_) and TOKEN_SHARE:
        corr = (sy_ - TOKEN_SHARE * py_) / (1 - TOKEN_SHARE)
        ax.plot(sx_, corr, ":", color="#c0504d", lw=1.4,
                label=f"same, prompt tokens excluded ({TOKEN_SHARE:.1%} of stream)")

    ax.set_ylabel("cross-entropy, nats/token")
    ax.set_title("Q10 pretraining: 25-layer, 10.0M params, batch 32, "
                 "constant LR 3e-4", fontsize=11, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9,
              bbox_to_anchor=(1.0, 0.80))
    ax.annotate("corpus rebuilt: prompt in every document, resume fixed\n"
                "(step 35,115; the loss dip is the new <|system|> tag "
                "being learned, see the bottom panel)",
                (CORPUS_SWITCH, 0.42), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-8, 0), ha="right",
                fontsize=9, color="#c0504d")

    # ---------------------------------------------------------- 2. val bpb
    ax = axes[1]
    vx, vy = column(both, "val_bpb_mixed")
    ax.plot(vx, vy, "o-", color="#1f4e79", ms=4, lw=1.6,
            label="val_bpb_mixed (held-out, bare)")
    ex, ey = column(both, "val_bpb_en")
    rx, ry = column(both, "val_bpb_ru")
    ax.plot(ex, ey, "--", color="#c0504d", lw=1.2, label="EN")
    ax.plot(rx, ry, "--", color="#4f81bd", lw=1.2, label="RU")
    ax.axhline(GPT2_BPB, color="#7f7f7f", ls=":", lw=1.3)
    ax.annotate(f"GPT-2-124M  {GPT2_BPB:.3f}", (vx[0], GPT2_BPB),
                textcoords="offset points", xytext=(4, 4),
                fontsize=9, color="#595959")
    ax.axhline(SOUP_BPB, color="#2e7d32", ls="-.", lw=1.3)
    ax.annotate(f"best+last soup  {SOUP_BPB:.3f}", (vx[0], SOUP_BPB),
                textcoords="offset points", xytext=(4, -12),
                fontsize=9, color="#2e7d32")
    ax.set_ylabel("bits per byte")
    ax.set_ylim(0.85, 1.62)
    ax.set_title("Held-out bpb (mixed eval slices, bytes counted per batch)",
                 fontsize=11, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.annotate("the mixed eval was added at step 24,000;\n"
                "the legacy val_bpb column is not plotted (different span)",
                (0.02, 0.60), xycoords="axes fraction", fontsize=8.5,
                color="#595959")

    # ---------------------------------------------------------- 3. prompt probe
    ax = axes[2]
    px, pn = column(sess, "nats_prompt")
    bx, bn = column(sess, "nats_bare")
    if len(px):
        ax.plot(px, smooth(pn, 9), color="#2e7d32", lw=1.6,
                label="loss on the ~21 tokens after <|system|>")
        ax.plot(bx, smooth(bn, 9), color="#7f7f7f", lw=1.6,
                label="loss on bare tokens in the same windows")
    ax.set_ylabel("cross-entropy, nats/token")
    ax.set_title("Prompt probe: the tag is learned, the text is not disturbed",
                 fontsize=11, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlabel("step")

    for ax in axes:
        ax.axvline(CORPUS_SWITCH, color="#c0504d", ls="--", lw=1.0, alpha=0.7)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].annotate("session 4", (0.90, 0.55), xycoords="axes fraction",
                     fontsize=9, color="#8a8a8a", ha="center")
    axes[2].annotate("session 4 only\n(the probe did not exist before)",
                     (0.02, 0.06), xycoords="axes fraction",
                     fontsize=8.5, color="#595959")

    # Shade the sessions so the 500-minute boundaries are visible.
    for start, end, name in [(1, CORPUS_SWITCH, "sessions 1-3"),
                             (CORPUS_SWITCH, int(x[-1]), "session 4")]:
        axes[0].axvspan(start, end, color="#1f4e79",
                        alpha=0.03 if name.endswith("4") else 0.06)

    fig.suptitle("KronLLM Q10 - pretraining curve to step 43,895 "
                 "(16.4 % of the epoch)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT, dpi=150)
    print(f"[out] {OUT}")

    # The numbers the plot is drawn from, so the figure can be checked.
    print(f"[train]  {len(x)} points, {x[0]}..{x[-1]}, "
          f"loss {y[:50].mean():.3f} -> {y[-50:].mean():.3f}")
    print(f"[val]    {len(vx)} mixed points: " +
          ", ".join(f"{a}:{b:.4f}" for a, b in zip(vx, vy)))


if __name__ == "__main__":
    main()
