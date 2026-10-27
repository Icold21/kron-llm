"""Plot the Q10 curve on the corrected validation metric.

The training log's ``val_bpb`` is measured on the first 512 blocks of the val
bin (English Python) and divided by the corpus-average 3.694 bytes/token, while
that span really has 2.56.  Two consequences for this plot:

* the logged curve can be **exactly** rescaled by ``3.694 / 2.56 = 1.444`` —
  same tokens, only the denominator fixed.  That is the green line, and it is
  the honest code-span curve for the whole history;
* the number that matters is the *mixed* one, measured by strides across the
  whole held-out set (``scripts/eval_reference.py``).  Those points arrive one
  per checkpoint, so the series starts with a single star and grows.

Reference lines come from GPT-2-124M scored on the same text.

Usage::

    python scripts/plot_q10.py [--live-step N]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = Path("data/kaggle_out/q10")
EPOCH_STEPS = 137_377
TOKENS_PER_STEP = 32 * 512
BPT_CORPUS = 3.6942873873459674
CODE_SPAN_JSON = Path("data/baseline_q10_first512.json")
EXPECTED = (0.95, 1.05)


def rows() -> list[dict]:
    path = RUN / "log.csv"
    if not path.exists():
        raise SystemExit(f"no {path}")
    return list(csv.DictReader(path.open(encoding="utf-8")))


def f(r: dict, key: str) -> float:
    try:
        return float(r.get(key) or "nan")
    except ValueError:
        return float("nan")


def rolling(xs: list[float], w: int) -> list[float]:
    out = []
    for i in range(len(xs)):
        chunk = xs[max(0, i - w + 1):i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def correction() -> float:
    """Factor that turns a logged code-span bpb into honest bits/byte."""
    if CODE_SPAN_JSON.exists():
        d = json.loads(CODE_SPAN_JSON.read_text(encoding="utf-8"))
        bpt = d["bytes"] / max(1, d["blocks"] * 512)
        if bpt > 0:
            return BPT_CORPUS / bpt
    return 1.444


def honest_points() -> list[dict]:
    """Mixed-domain measurements: one per downloaded checkpoint."""
    out = []
    for name in glob.glob("data/baseline_q10_*.json"):
        if "first" in name:
            continue
        d = json.loads(Path(name).read_text(encoding="utf-8"))
        if d.get("step"):
            out.append(d)
    return sorted(out, key=lambda d: d["step"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/q10_curve.png")
    ap.add_argument("--live-step", type=int, default=0)
    ap.add_argument("--live-val", type=float, default=0.0,
                    help="last val bpb read from the live session log, if any")
    args = ap.parse_args()

    data = rows()
    train = [(int(r["step"]), f(r, "loss")) for r in data if r["split"] == "train"]
    val = [(int(r["step"]), f(r, "val_bpb")) for r in data
           if r["split"] == "val" and math.isfinite(f(r, "val_bpb"))]
    live_step = args.live_step or (int(train[-1][0]) if train else 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150,
                                   sharex=False)
    fig.patch.set_facecolor("white")

    # --- training loss -----------------------------------------------------
    xs = [s for s, v in train if math.isfinite(v)]
    ys = [v for _, v in train if math.isfinite(v)]
    ax1.plot(xs, ys, color="#9ec5e8", lw=0.8, alpha=0.9,
             label="train loss (каждые 100 шагов)")
    ax1.plot(xs, rolling(ys, 25), color="#0b62c4", lw=1.8,
             label="скользящая медиана (25 точек)")
    ax1.axvline(live_step, color="#d1495b", ls=":", lw=1.2)
    ax1.annotate(f"живой шаг {live_step:,}", xy=(live_step, max(ys)),
                 xytext=(live_step * 0.5, max(ys) * 0.97),
                 fontsize=8, color="#d1495b")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_xlabel("шаг")
    ax1.set_title("Q10: кривая обучения", fontsize=11, loc="left")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # --- validation --------------------------------------------------------
    k = correction()
    vx = [s for s, _ in val]
    vy = [v for _, v in val]
    ax2.plot(vx, vy, ":", color="#b0b0b0", lw=1.2,
             label=f"старый протокол (делит на {BPT_CORPUS:.2f} б/ток)")
    ax2.plot(vx, [v * k for v in vy], "-o", color="#1b6b3a", ms=3.5, lw=1.5,
             label=f"код, честные байты (×{k:.2f})")

    pts = honest_points()
    if pts:
        ax2.plot([p["step"] for p in pts], [p["bpb"] for p in pts], "*",
                 ms=13, color="#d1495b",
                 label="весь бин, страйд (EN+RU, все домены)")
        last = pts[-1]
        ax2.annotate(f"{last['bpb']:.3f}", xy=(last["step"], last["bpb"]),
                     xytext=(10, -14), textcoords="offset points",
                     fontsize=9, color="#d1495b")

    gpt2 = Path("data/baseline_gpt2.json")
    if gpt2.exists():
        g = json.loads(gpt2.read_text(encoding="utf-8"))
        ax2.axhline(g["bpb"], color="#7b3fa0", ls="--", lw=1.1,
                    label=f"GPT-2 124M, тот же страйд ({g['bpb']:.3f})")
    gpt2_code = Path("data/baseline_gpt2_first.json")
    if gpt2_code.exists():
        g = json.loads(gpt2_code.read_text(encoding="utf-8"))
        ax2.axhline(g["bpb"], color="#7b3fa0", ls=":", lw=1.0, alpha=0.6,
                    label=f"GPT-2 124M, код ({g['bpb']:.3f})")

    ax2.axhspan(*EXPECTED, color="#f2c14e", alpha=0.22, lw=0,
                label=f"ориентир к концу эпохи {EXPECTED[0]:.2f}–{EXPECTED[1]:.2f}")
    ax2.axvline(EPOCH_STEPS, color="#444", ls=":", lw=1)
    ax2.annotate("конец эпохи\n137,377 шагов",
                 xy=(EPOCH_STEPS, EXPECTED[1]), xytext=(0.30, 0.72),
                 textcoords="axes fraction", fontsize=8, color="#444",
                 arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))

    ax2.set_ylabel("bits per byte")
    ax2.set_xlabel("шаг (лог. шкала)")
    ax2.set_xscale("log")
    ax2.set_xlim(900, 220_000)
    ax2.set_xticks([1_000, 3_000, 10_000, 30_000, 100_000])
    ax2.set_xticklabels(["1k", "3k", "10k", "30k", "100k"])
    ax2.set_title("Q10: валидация — честная шкала (held-out 2 %, 46.2M токенов)",
                  fontsize=11, loc="left")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="upper right")

    fig.text(0.01, 0.005,
             f"шаг {live_step:,} из {EPOCH_STEPS:,} "
             f"({100*live_step/EPOCH_STEPS:.1f} % эпохи) · "
             f"{live_step*TOKENS_PER_STEP/1e9:.2f}B токенов · "
             f"2.53 с/шаг · single T4 · страйдовые точки — с чекпоинта сессии 1",
             fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(out)
    print(f"[plot] code-span correction ×{k:.3f}; "
          f"{len(pts)} mixed points; live step {live_step:,}")


if __name__ == "__main__":
    main()
