"""Publication figure: val bpb across the whole Q10 pretraining run.

Different from ``plot_curve.py``, which is the internal diagnostic (Russian
labels, prompt probe, shaded session bands).  This one is meant to be dropped
into an article or a model card:

* the honest metric -- ``val_bpb_mixed`` on the held-out bilingual bin, which is
  comparable across every session because the val bin never changed;
* before step 24,000 only the legacy metric exists (one English slice divided by
  the corpus-wide bytes-per-token), and it is plotted dotted after multiplying by
  1.444, the factor that puts it on the same scale; at step 35,000 the two agree
  (0.813 x 1.444 = 1.174 against 1.180 measured);
* the x axis in training tokens, not steps, so the Chinchilla comparison is
  legible (Q10 reaches ~1.46B tokens, about 147 tokens per parameter);
* the GPT-2-124M reference as a horizontal line, the only number a reader can
  anchor on;
* the two phases shaded: the epoch pass, stopped at 58,213 when 23k further steps
  had bought 0.0003 bpb, and the decay pass with its 3e-4 -> 1e-6 schedule.

Sessions 9 and 10 hold the same run (10 is a copy), so steps are de-duplicated:
the first value seen for a step wins.

Usage::

    python scripts/plot_paper_curve.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "q10_history.csv"
SESSIONS = Path(r"E:/kron_out")
OUT_PNG = ROOT / "data" / "fig_pretrain_curve.png"
OUT_PDF = ROOT / "data" / "fig_pretrain_curve.pdf"

TOKENS_PER_STEP = 32 * 512          # batch 32 x seq 512
LEGACY_SCALE = 1.444                # see docs/results.md: same span, right divisor
EPOCH_END = 58_213
DECAY_END = 109_507
REACHED = 88_936
GPT2 = 1.554


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def series() -> tuple[dict[int, float], dict[int, float]]:
    """(mixed, legacy-rescaled) as step -> bpb, from history and session logs."""
    mixed: dict[int, float] = {}
    legacy: dict[int, float] = {}
    for path in [HISTORY] + sorted(SESSIONS.glob("session*/**/log.csv")):
        try:
            data = rows(path)
        except OSError:
            continue
        for r in data:
            if r.get("split") != "val":
                continue
            step = int(r["step"])
            m = (r.get("val_bpb_mixed") or "").strip()
            if m:
                mixed.setdefault(step, float(m))
                continue
            v = (r.get("val_bpb") or "").strip()
            if v:
                legacy.setdefault(step, float(v) * LEGACY_SCALE)
    # a step with a measured mixed value does not also need its legacy twin
    legacy = {s: v for s, v in legacy.items() if s not in mixed}
    return dict(sorted(mixed.items())), dict(sorted(legacy.items()))


def main() -> None:
    mixed, legacy = series()
    xs = list(mixed)
    print(f"mixed {len(xs)} points {xs[0]:,}..{xs[-1]:,} | "
          f"legacy {len(legacy)} points")

    def gb(step: int) -> float:
        return step * TOKENS_PER_STEP / 1e9

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=300, constrained_layout=True)
    ax.axvspan(0, gb(EPOCH_END), color="#eef2f7", zorder=0)
    ax.axvspan(gb(EPOCH_END), gb(DECAY_END), color="#fdf1e7", zorder=0)

    ax.plot([gb(s) for s in legacy], list(legacy.values()),
            color="#8a8a8a", lw=1.2, ls=":", zorder=2,
            label="earlier checkpoints (legacy metric, rescaled x1.444)")
    ax.plot([gb(s) for s in xs], list(mixed.values()),
            color="#1f4e79", lw=1.9, marker="o", ms=3, zorder=3,
            label="Q10 (9.9M params), held-out bits per byte")
    ax.axhline(GPT2, color="#b03a2e", ls="--", lw=1.2, zorder=2,
               label="GPT-2 124M, same text and metric: 1.554")
    for step in (EPOCH_END, REACHED):
        ax.axvline(gb(step), color="#7f8c8d", lw=0.9, ls=":", zorder=1)

    # the last checkpoint of the epoch pass, not a hard-coded value
    epoch_step = max(s for s in xs if s <= EPOCH_END)
    ax.annotate(f"epoch pass stopped at {mixed[epoch_step]:.3f}:\n"
                f"23k more steps bought 0.0003",
                xy=(gb(epoch_step), mixed[epoch_step]),
                xytext=(gb(EPOCH_END) - 0.62, 1.34), fontsize=6.5, color="#4a4a4a",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#7f8c8d"))
    ax.annotate("decay pass, lr 3e-4 -> 1e-6",
                xy=(gb(72_000), 1.30), xytext=(gb(74_000), 1.40),
                fontsize=6.5, color="#8a5a2b")
    ax.annotate(f"{mixed[xs[-1]]:.3f} at step {xs[-1]:,}",
                xy=(gb(xs[-1]), mixed[xs[-1]]),
                xytext=(gb(xs[-1]) - 0.55, 1.10), fontsize=6.5, color="#1f4e79",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#1f4e79"))

    ax.set_xlabel("Training tokens (billions)", fontsize=8.5)
    ax.set_ylabel("Held-out bits per byte", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(0, gb(DECAY_END))
    ax.set_ylim(1.05, 2.15)
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print("written", OUT_PNG, "and", OUT_PDF)


if __name__ == "__main__":
    main()
