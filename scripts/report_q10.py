"""Session / pretrain report for Q10: the loss curve, and how good it is.

The Kaggle API only exposes a run's artefacts after it stops, so the chain
script downloads ``log.csv`` at every session end and this script turns it into
the report that goes by email.  It keeps a merged history across sessions, so
the curve is continuous from step 1 even though it arrives in pieces.

Assessment is deliberately mechanical: a power-law fit on the last part of the
curve projects the end of the epoch, and the verdict is a set of explicit
thresholds, so the note does not silently drift into wishful reading.

Usage::

    python scripts/report_q10.py --tag session
    python scripts/report_q10.py --tag final
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

RUN = Path("data/kaggle_out/q10")
HIST = Path("data/q10_history.csv")
OUT = Path("data/report_q10.md")

EPOCH_STEPS = 137_377
TOKENS_PER_STEP = 32 * 512


def load_rows() -> list[dict]:
    """Union of this session's log and everything seen before."""
    rows: dict[tuple[int, str], dict] = {}
    if HIST.exists():
        for r in csv.DictReader(HIST.open(encoding="utf-8")):
            rows[(int(r["step"]), r["split"])] = r
    src = RUN / "log.csv"
    if src.exists():
        for r in csv.DictReader(src.open(encoding="utf-8")):
            rows[(int(r["step"]), r["split"])] = r
    merged = sorted(rows.values(), key=lambda r: (int(r["step"]), r["split"]))
    if merged:
        HIST.parent.mkdir(parents=True, exist_ok=True)
        with HIST.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(merged[0]))
            w.writeheader()
            w.writerows(merged)
    return merged


def fnum(r: dict, key: str) -> float:
    try:
        v = r.get(key) or ""
        return float(v)
    except ValueError:
        return float("nan")


def milestone_table(train: list[dict]) -> tuple[str, list[dict]]:
    wanted = [1, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000,
              100000]
    picked: list[dict] = []
    by_step = {int(r["step"]): r for r in train}
    for w in wanted:
        if w in by_step:
            picked.append(by_step[w])
    if train and int(train[-1]["step"]) not in {int(r["step"]) for r in picked}:
        picked.append(train[-1])
    lines = ["| step | tokens | loss | bpb | sec/step |", "|---:|---:|---:|---:|---:|"]
    for r in picked:
        s = int(r["step"])
        lines.append(f"| {s:,} | {s * TOKENS_PER_STEP / 1e6:.0f}M | "
                     f"{fnum(r, 'loss'):.3f} | {fnum(r, 'bpb'):.3f} | "
                     f"{fnum(r, 'sec_per_step'):.2f} |")
    return "\n".join(lines), picked


def smooth(rows: list[dict], window: int = 5) -> list[tuple[float, float]]:
    xs, out = [], []
    for r in rows:
        b = fnum(r, "bpb")
        if math.isfinite(b):
            xs.append((int(r["step"]), b))
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = [b for _, b in xs[lo:i + 1]]
        out.append((xs[i][0], sum(chunk) / len(chunk)))
    return out


def power_fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """bpb = a * step^-b by least squares in log-log space."""
    pts = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(pts) < 8:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    den = sum((v - mx) ** 2 for v in lx)
    if den <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / den
    a = math.exp(my - b * mx)
    return a, -b


def verdict(bpb: float) -> str:
    if bpb <= 0.85:
        return ("**сильный результат**: ниже 0.85 bpb на смешанном EN/RU для "
                "10M-модели — это уровень, на котором Kronecker-параметризация "
                "уже сама по себе является аргументом")
    if bpb <= 0.95:
        return ("**крепкий результат**: 0.85–0.95 bpb — ровно то, что должен "
                "показывать здоровый 10M-трансформер, обученный сильно за "
                "пределами compute-optimal")
    if bpb <= 1.05:
        return ("**в норме, но без запаса**: 0.95–1.05 bpb. Смотреть не на "
                "абсолют, а на наклон: если кривая на финише всё ещё падает, "
                "модель не выбрала ёмкость")
    return ("**ниже ожиданий**: выше 1.05 bpb для 2.25B токенов — повод "
            "разбираться (проверить val-состав, LR, нормы весов)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="session",
                    choices=["session", "final", "status"])
    args = ap.parse_args()

    rows = load_rows()
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train:
        OUT.write_text("# Q10: данных пока нет\n\n`log.csv` ещё не скачан.\n",
                       encoding="utf-8")
        print(OUT)
        return

    last = train[-1]
    step = int(last["step"])
    sec = statistics.median([fnum(r, "sec_per_step") for r in train[-50:]
                             if math.isfinite(fnum(r, "sec_per_step"))] or [0])
    tps = TOKENS_PER_STEP / sec if sec else 0
    tok_seen = step * TOKENS_PER_STEP
    pct = 100 * step / EPOCH_STEPS
    remain_s = (EPOCH_STEPS - step) * sec
    gpu_h = remain_s / 3600
    sessions = gpu_h / 8.33

    table, picked = milestone_table(train)
    curve = smooth(train[-400:])
    # Validation bpb is cleaner than the batch losses, so prefer it when there
    # are enough points; fall back to the smoothed training curve early on.
    val_pts = [(int(r["step"]), fnum(r, "val_bpb")) for r in val]
    val_pts = [(x, y) for x, y in val_pts if math.isfinite(y) and y > 0]
    fit = power_fit(val_pts) if len(val_pts) >= 8 else (
        power_fit(curve[-200:]) if len(curve) >= 8 else None)
    projected = None
    if fit:
        a, b = fit
        projected = a * EPOCH_STEPS ** (-b)

    head = {
        "session": "Отчёт по сессии Q10",
        "status": "Статус Q10",
        "final": "Q10: конец претрейна",
    }[args.tag]

    L = [f"# {head}", ""]
    L += [
        f"- шаг **{step:,}** из {EPOCH_STEPS:,} ({pct:.1f} % эпохи)",
        f"- токенов пройдено **{tok_seen / 1e9:.3f}B**, скорость {tps:,.0f} ток/с "
        f"({sec:.2f} с/шаг)",
        f"- loss {fnum(last, 'loss'):.3f}, bpb {fnum(last, 'bpb'):.3f}",
    ]
    if val:
        lv = val[-1]
        L.append(f"- **val bpb {fnum(lv, 'val_bpb'):.4f}** на шаге {lv['step']}")
    L += [
        f"- до конца эпохи ~**{gpu_h:.1f} GPU-ч** ≈ {sessions:.1f} сессий "
        f"(~{sessions * 8.5 / 24:.1f} суток календаря)",
        "",
        "## Кривая",
        "",
        table,
    ]

    if val:
        L += ["", "## Валидация", "", "| шаг | val loss | val bpb |",
              "|---:|---:|---:|"]
        for r in val[-10:]:
            L.append(f"| {r['step']} | {fnum(r, 'val_loss'):.3f} | "
                     f"{fnum(r, 'val_bpb'):.4f} |")

    if fit:
        L += ["", "## Оценка", "",
              f"Степенная подгонка по последним точкам: bpb = "
              f"{fit[0]:.3f} · step^(−{fit[1]:.3f}). При таком наклоне к шагу "
              f"{EPOCH_STEPS:,} выходит **~{projected:.3f} bpb**.", ""]
        if pct < 25:
            L += [f"Осторожно: подгонка сделана на первых {pct:.1f} % прогона. "
                  "На таком отрезке кривая ещё крутая, и экстраполяция через "
                  "всю эпоху систематически оптимистична — реальный наклон "
                  "выполаживается. Число выше читать как верхнюю границу, а не "
                  "как прогноз.", ""]
        current = fnum(last, "bpb")
        L += ["Ориентиры, чтобы число не висело в пустоте:", "",
              "| уровень | bpb |", "|---|---:|",
              "| равномерное распределение по байтам | 8.0 |",
              "| gzip на тексте | ~2.0–2.5 |",
              "| символьные n-граммы | ~1.4–1.6 |",
              "| GPT-2-класс (124M) на англ. вебе | ~1.0 |",
              f"| **наша кривая сейчас** | **{current:.3f}** |",
              f"| **прогноз на конец эпохи** | **~{projected:.3f}** |", "",
              f"Это соответствует сжатию текста в **{8 / max(current, 1e-9):.1f}×** "
              f"против сырых байтов (xz на тексте даёт ~5–6×).", ""]
        L += [f"Вердикт по прогнозу: {verdict(projected)}.", ""]

    L += [
        "## Что дальше",
        "",
        "Эпоха идёт на константном LR (WSD). После её конца — decay-проход "
        "20 % эпохи (27,475 шагов, ~19 GPU-ч) на отобранных данных, LR "
        "3e-4 → 1e-6. Именно он, а не конец эпохи, даёт финальное качество: "
        "чекпоинт конца эпохи останется «горячим».",
    ]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    # Subject hint and the "epoch finished" flag for the chain script.
    val_bpb = fnum(val[-1], "val_bpb") if val else float("nan")
    if step >= EPOCH_STEPS:
        subject = (f"Q10: эпоха завершена — val bpb {val_bpb:.3f}, "
                   f"{tok_seen / 1e9:.2f}B токенов")
        Path("data/q10_epoch_done.flag").write_text(str(step), encoding="utf-8")
    else:
        subject = (f"Q10 {pct:.0f}% эпохи: step {step:,}, bpb "
                   f"{fnum(last, 'bpb'):.3f}"
                   + (f", val {val_bpb:.3f}" if math.isfinite(val_bpb) else ""))
    Path("data/report_q10_subject.txt").write_text(subject, encoding="utf-8")
    print(OUT)
    print(f"[report] step {step:,} ({pct:.1f} %), val bpb {val_bpb:.4f}")
    print(f"[subject] {subject}")


if __name__ == "__main__":
    main()
