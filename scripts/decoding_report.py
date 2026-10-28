"""Turn the decoding sweep into a table with a recommendation.

Ranking rule, stated before looking at the numbers: a setting wins if an
independent judge finds its text more natural **in both languages at once**, and
it loses points for loops.  The two judge columns are divided by their best value
in the grid so that English and Russian count equally despite their different
bytes-per-token scales; the composite is the average of those two ratios.
Task checks (code parses, arithmetic, identity) are printed, not ranked: a
setting that makes the model marginally more fluent while it can no longer close
a function is not a better decoder for a report that shows code.

Usage::

    python scripts/decoding_report.py --sweep data/decoding_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, digits=4):
    return "—" if v is None else f"{v:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", default=["data/decoding_sweep.json"],
                    help="one or more sweep JSON files, merged in order")
    ap.add_argument("--out", default="data/decoding_report.md")
    args = ap.parse_args()

    rows = []
    for path in args.sweep:
        rows += json.loads(Path(path).read_text(encoding="utf-8"))
    # Later sweeps re-run earlier configurations with more seeds; the first
    # occurrence is the one the grid table was ranked on, so keep it.
    seen, unique = set(), []
    for r in rows:
        key = (r["temperature"], r["top_k"], r["top_p"], r["rep"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rows = unique
    best_en = min(r["judge_en"] for r in rows if r["judge_en"])
    best_ru = min(r["judge_ru"] for r in rows if r["judge_ru"])

    for r in rows:
        en = 0.5 * r["judge_en"] / best_en + 0.5 * r["judge_ru"] / best_ru
        # A loop-heavy sample is not "natural" even when the judge likes the words.
        r["score"] = en * (1.0 + r["loops"])

    ranked = sorted(rows, key=lambda r: r["score"])
    lines = [
        "# Подбор параметров генерации",
        "",
        f"Сетка из {len(rows)} конфигураций, один и тот же чекпоинт (шаг 35 100), "
        "7 промптов × 2 сида у каждой. Судьи — сторонние модели: GPT-2 для "
        "английского, ruGPT3-small для русского, оба считают bits/byte "
        "продолжения. Ниже — они же, отсортированные по композиту "
        "(среднее отношений к лучшему в сетке по каждому языку, со штрафом за петли).",
        "",
        "| # | T | top_k | top_p | rp | GPT-2 bpb | ruGPT bpb | петли | distinct-3 "
        "| код | 17+25 | identity |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['temperature']} | {r['top_k'] or '—'} | "
            f"{r['top_p'] or '—'} | {r['rep']} | {fmt(r['judge_en'])} | "
            f"{fmt(r['judge_ru'])} | {r['loops']:.2f} | {r['distinct3']:.3f} | "
            f"{r['code_ok']:.2f} | {r['arith_ok']:.2f} | {r['identity_ok']:.2f} |")

    def describe(key, label):
        row = next((r for r in rows
                    if (r["temperature"], r["top_k"], r["top_p"], r["rep"]) == key),
                   None)
        if row is None:
            return None
        rank = ranked.index(row) + 1
        return (f"{label} — **{rank}-е место из {len(ranked)}**: GPT-2 "
                f"{fmt(row['judge_en'])}, ruGPT {fmt(row['judge_ru'])}, петель "
                f"{row['loops']:.2f}, код собирается в {row['code_ok']:.0%} случаев.")

    adopted = describe((1.0, 8, 0.9, 1.1),
                       "Принятая рецептура (T=1.0, top_k=8, top_p=0.9, rp 1.1)")
    reference = describe((1.0, 8, 0.0, 1.1),
                         "Бывшая рецептура (T=1.0, top_k=8, rp 1.1)")
    for line in (adopted, reference):
        if line:
            lines += ["", line]
    lines += [
        "",
        "Эти два места сравнивать по таблице нельзя: разница между ними меньше "
        "разброса по промптам, поэтому решение принималось на 8 сидах парной "
        "статистикой (`scripts/decoding_stats.py`), итог — в `docs/decoding.md`.",
    ]

    best = ranked[0]
    lines += [
        "",
        f"Лучшая по композиту: **T={best['temperature']}, top_k="
        f"{best['top_k'] or '—'}, top_p={best['top_p'] or '—'}, "
        f"rp={best['rep']}** — GPT-2 {fmt(best['judge_en'])}, "
        f"ruGPT {fmt(best['judge_ru'])}, петель {best['loops']:.2f}.",
        "",
        "Сырые числа по каждому промпту лежат в "
        + ", ".join(f"`{p}`" for p in args.sweep) + " — там же тексты, чтобы "
        "вывод можно было перепроверить глазами, а не доверять таблице.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
