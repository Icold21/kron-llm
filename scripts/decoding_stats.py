"""Are the differences between decoding settings real, or seed noise?

The sweep stores the generated texts, so the judges can be re-run on exactly the
same pairs.  That makes the comparison paired: configuration A and configuration
B are scored on the same prompt with the same seed, and only the difference is
looked at.  Paired differences remove the prompt-to-prompt spread, which is much
larger than the effect being measured.

Reports mean judge bpb per language with its standard error, and the paired
difference against a reference configuration with the error of that difference.
A difference smaller than twice its own error is not a finding.

Usage::

    python scripts/decoding_stats.py --reference 1.0/8/0.0/1.1 \
        --pick 1.0/8/0.0/1.05 1.0/8/0.0/1.1 0.8/8/0.0/1.1 1.0/0/0.95/1.1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT_LANG = {
    "en_prose": "en", "en_science": "en",
    "ru_prose": "ru", "ru_history": "ru",
}
PROMPT_TEXT = {
    "en_prose": "She opened the door and",
    "en_science": "The Moon orbits the Earth at a distance of",
    "ru_prose": "Он открыл дверь и",
    "ru_history": "В 1961 году",
}


def load_judge_class():
    spec = importlib.util.spec_from_file_location(
        "tune_decoding", ROOT / "scripts" / "tune_decoding.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Judge


def key_of(row: dict) -> str:
    return f"{row['temperature']}/{row['top_k']}/{row['top_p']}/{row['rep']}"


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return m, math.sqrt(var / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+",
                    default=["data/decoding_sweep.json",
                             "data/decoding_sweep_b.json"])
    ap.add_argument("--reference", default="1.0/8/0.0/1.1")
    ap.add_argument("--pick", nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for path in args.sweep:
        p = Path(path)
        if p.exists():
            rows += json.loads(p.read_text(encoding="utf-8"))
    by_key = {key_of(r): r for r in rows}
    # The reference has to be scored too, otherwise there is nothing to subtract.
    picks = list(dict.fromkeys([args.reference] + args.pick))

    Judge = load_judge_class()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    judges = {"en": Judge("gpt2", device),
              "ru": Judge("ai-forever--rugpt3small_based_on_gpt2".replace("--", "/"),
                          device)}

    scored: dict[str, dict[str, dict[tuple[str, int], float]]] = {}
    for want in picks:
        row = by_key.get(want)
        if row is None:
            print(f"[skip] {want} not in sweep")
            continue
        table: dict[str, dict[tuple[str, int], float]] = {"en": {}, "ru": {}}
        for label, texts in row["samples"].items():
            lang = PROMPT_LANG.get(label)
            if lang is None:
                continue
            for i, text in enumerate(texts):
                value = judges[lang].bpb(PROMPT_TEXT[label], text)
                if value is not None:
                    table[lang][(label, i)] = value
        scored[want] = table

    ref = scored.get(args.reference)
    print(f"\n### paired against {args.reference}\n")
    print("| config | lang | judge bpb | ±SE | Δ vs ref | ±SE(Δ) | verdict |")
    print("|---|---|---:|---:|---:|---:|---|")
    for want, table in scored.items():
        for lang in ("en", "ru"):
            cells = sorted(table[lang])
            vals = [table[lang][c] for c in cells]
            m, se = mean_se(vals)
            if ref is not None and want != args.reference:
                pairs = [(table[lang][c], ref[lang][c]) for c in cells
                         if c in ref[lang]]
                diffs = [a - b for a, b in pairs]
                d, dse = mean_se(diffs)
                verdict = ("лучше" if d < 0 else "хуже") if abs(d) > 2 * dse else "в пределах шума"
                print(f"| {want} | {lang} | {m:.4f} | {se:.4f} | {d:+.4f} | "
                      f"{dse:.4f} | {verdict} |")
            else:
                print(f"| {want} | {lang} | {m:.4f} | {se:.4f} | — | — | — |")


if __name__ == "__main__":
    main()
