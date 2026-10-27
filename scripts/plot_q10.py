"""Plot the Q10 curve for the email reports.

The email body is HTML because tables read badly as Markdown, but a curve is
the one thing a table cannot replace: the shape (fast early descent, slow
logarithmic tail) is the whole argument about whether the run is healthy.  So
the report gets a PNG attachment.

Left panel is the training loss as logged (noisy, 100-step points, plus a
rolling median); right panel is validation bpb against the expected band for
the end of the epoch.  Deliberately no extrapolated curve: a power law fitted
to the first 8 % of the steps projects 0.56 bpb, which is arithmetic, not a
forecast.  The band is an expectation from comparable runs, and the plot says
so.

Usage::

    python scripts/plot_q10.py            # data/q10_curve.png

Live session-2 points are appended by hand from the session log stream: the
Kaggle API only hands over log.csv after a run ends.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = Path("data/kaggle_out/q10")
EPOCH_STEPS = 137_377
TOKENS_PER_STEP = 32 * 512

# Points read from the live session log (step, val bpb).  Kept explicit and
# labelled as live so the plot never pretends a stream read is a downloaded
# curve.
LIVE = [(13_000, 0.870)]


def rows() -> list[dict]:
    path = RUN / "log.csv"
    if not path.exists():
        raise SystemExit(f"no {path}")
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        out.append(r)
    return out


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


def power_fit(pts: list[tuple[float, float]]) -> tuple[float, float] | None:
    pts = [(x, y) for x, y in pts if x > 0 and y > 0]
    if len(pts) < 8:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    den = sum((v - mx) ** 2 for v in lx)
    if den <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / den
    return math.exp(my - b * mx), -b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/q10_curve.png")
    ap.add_argument("--live-step", type=int, default=14_250)
    args = ap.parse_args()

    data = rows()
    train = [(int(r["step"]), f(r, "loss")) for r in data if r["split"] == "train"]
    val = [(int(r["step"]), f(r, "val_bpb")) for r in data
           if r["split"] == "val" and math.isfinite(f(r, "val_bpb"))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), dpi=150,
                                   sharex=False)
    fig.patch.set_facecolor("white")

    # --- training loss -----------------------------------------------------
    xs = [s for s, v in train if math.isfinite(v)]
    ys = [v for _, v in train if math.isfinite(v)]
    ax1.plot(xs, ys, color="#9ec5e8", lw=0.8, alpha=0.9,
             label="train loss (каждые 100 шагов)")
    ax1.plot(xs, rolling(ys, 25), color="#0b62c4", lw=1.8,
             label="скользящая медиана (25 точек)")
    ax1.axvline(args.live_step, color="#d1495b", ls=":", lw=1.2)
    ax1.annotate(f"живой шаг {args.live_step:,}", xy=(args.live_step, max(ys)),
                 xytext=(args.live_step * 0.55, max(ys) * 0.98),
                 fontsize=8, color="#d1495b")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_xlabel("шаг")
    ax1.set_title("Q10: кривая обучения (сессия 1 скачана, сессия 2 идёт)",
                  fontsize=11, loc="left")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # --- validation bpb ----------------------------------------------------
    vx = [s for s, v in val]
    vy = [v for _, v in val]
    ax2.plot(vx, vy, "o-", color="#1b6b3a", ms=4, lw=1.4, label="val bpb")
    if LIVE:
        lx_, ly_ = zip(*LIVE)
        ax2.plot(lx_, ly_, "*", ms=11, color="#d1495b",
                 label="val bpb (живой лог сессии 2)")
    fit = power_fit(list(zip(vx, vy)))
    projected = None
    if fit:
        a, b = fit
        projected = a * EPOCH_STEPS ** (-b)
    ax2.axvspan(0, EPOCH_STEPS, ymin=0, ymax=0)
    # The expectation band, not a projection: a fitted power law on the first
    # 8 % of the run gives 0.56 bpb, which would be a lie dressed as a forecast.
    ax2.axhspan(0.80, 0.83, color="#f2c14e", alpha=0.25, lw=0,
                label="ожидаемый диапазон к концу эпохи")
    ax2.axvline(EPOCH_STEPS, color="#444", ls=":", lw=1)
    ax2.annotate("конец эпохи\n137,377 шагов",
                 xy=(EPOCH_STEPS, 0.83), xytext=(0.42, 0.62),
                 textcoords="axes fraction", fontsize=8, color="#444",
                 arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))
    ax2.annotate(f"живой шаг {args.live_step:,}", xy=(args.live_step, LIVE[-1][1]),
                 xytext=(0.30, 0.30), textcoords="axes fraction",
                 fontsize=8, color="#d1495b",
                 arrowprops=dict(arrowstyle="->", color="#d1495b", lw=0.8))
    ax2.set_ylabel("validation bits per byte")
    ax2.set_xlabel("шаг (лог. шкала)")
    # Log x: 1k -> 137k on one axis, otherwise the whole curve crushes into the
    # left edge and the flattening (the thing worth looking at) disappears.
    ax2.set_xscale("log")
    ax2.set_xlim(900, 200_000)
    ax2.set_xticks([1_000, 3_000, 10_000, 30_000, 100_000])
    ax2.set_xticklabels(["1k", "3k", "10k", "30k", "100k"])
    ax2.set_title("Q10: валидация (held-out 2 %, EN+RU)", fontsize=11, loc="left")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    last_val = LIVE[-1] if LIVE else (vx[-1], vy[-1])
    ax2.annotate(f"{last_val[1]:.3f}", xy=last_val, xytext=(8, 6),
                 textcoords="offset points", fontsize=9, color="#d1495b")

    fig.text(0.01, 0.005,
             f"шаг {args.live_step:,} из {EPOCH_STEPS:,} "
             f"({100*args.live_step/EPOCH_STEPS:.1f} % эпохи) · "
             f"{args.live_step*TOKENS_PER_STEP/1e9:.2f}B токенов · "
             f"2.53 с/шаг · single T4",
             fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(out)
    if projected:
        print(f"[plot] projected end-of-epoch bpb ~{projected:.3f}")


if __name__ == "__main__":
    main()
