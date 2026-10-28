"""Side-by-side generations: last.pt versus the averaged weights.

Both runs used the same prompts and the same seeds, so sample ``i`` of a prompt
is the same draw from a different set of weights -- a paired comparison a reader
can judge by eye, not just by the judge's bpb.

Usage::

    python scripts/soup_examples.py --seeds 0,1,2 --out data/soup_examples.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "en_prose": "She opened the door and",
    "en_science": "The Moon orbits the Earth at a distance of",
    "ru_prose": "Он открыл дверь и",
    "ru_history": "В 1961 году",
    "code": "def add(a, b):\\n    return",
    "arith": "17 + 25 =",
    "identity": "KronLLM is",
}


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    row = data["samplers"][0]
    return {"samples": row["samples"], "row": row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/free_gains_gen_last.json")
    ap.add_argument("--b", default="data/free_gains_gen_soup.json")
    ap.add_argument("--label-a", default="last.pt (шаг 35 115)")
    ap.add_argument("--label-b", default="soup (50/50, шаг 35 000 + 35 115)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="data/soup_examples.md")
    args = ap.parse_args()

    run_a, run_b = load(args.a), load(args.b)
    seeds = [int(s) for s in args.seeds.split(",")]
    lines = [
        "# Генерация: `last.pt` против усреднённых весов",
        "",
        "Одинаковые промпты, одинаковые сиды, рецепт T = 1.0, top_k = 8, "
        "top_p = 0.9, rp = 1.1. Номер сида — это один и тот же бросок кубика "
        "для обеих моделей, поэтому пары сравнимы построчно.",
        "",
        "| | bpb судьи EN | bpb судьи RU | циклы | distinct-3 |",
        "|---|---:|---:|---:|---:|",
        f"| {args.label_a} | {run_a['row']['judge_en']:.3f} | "
        f"{run_a['row']['judge_ru']:.3f} | {run_a['row']['loops']:.2f} | "
        f"{run_a['row']['distinct3']:.3f} |",
        f"| {args.label_b} | {run_b['row']['judge_en']:.3f} | "
        f"{run_b['row']['judge_ru']:.3f} | {run_b['row']['loops']:.2f} | "
        f"{run_b['row']['distinct3']:.3f} |",
        "",
    ]
    for label, prompt in LABELS.items():
        lines += [f"## `{label}` — промпт `{prompt}`", ""]
        for seed in seeds:
            text_a = run_a["samples"][label][seed]
            text_b = run_b["samples"][label][seed]
            lines += [
                f"**сид {seed}**",
                "",
                f"*{args.label_a}:*",
                "",
                "```",
                text_a,
                "```",
                "",
                f"*{args.label_b}:*",
                "",
                "```",
                text_b,
                "```",
                "",
            ]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
