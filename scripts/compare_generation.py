"""Paired comparison of two checkpoints on the same generation probes.

Both runs use the same prompts and the same seeds, so the honest statistic is
the *paired* difference: for every (prompt, seed) pair subtract one run from the
other, then average.  That removes the prompt-to-prompt variance, which is the
dominant term and the reason unpaired numbers on seven prompts disagree with
each other between runs.

The generation JSONs store the sampled texts, so the paired statistics that need
only text (loops, distinct-3, code parses, length) come free.  Judge bpb needs
the 125M judges, so it is recomputed here unless ``--no-judge`` is passed.

Usage::

    python scripts/compare_generation.py data/free_gains_gen_last.json \
        data/free_gains_gen_soup.json --label-a last --label-b soup \
        --out data/free_gains_paired.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from free_gains import (PROMPTS, Judge, code_parses,  # noqa: E402
                        distinct3, loop_rate)


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["samplers"][0]
    return {"path": str(path), "data": data, "row": row,
            "samples": row["samples"]}


def paired(values_a: list, values_b: list) -> dict:
    deltas = [b - a for a, b in zip(values_a, values_b)
              if a is not None and b is not None]
    n = len(deltas)
    if n < 2:
        return {"n": n, "delta": None}
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(var / n)
    return {"n": n, "delta": mean, "se": se,
            "sigma": (abs(mean) / se) if se else float("inf")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_a, run_b = load(Path(args.a)), load(Path(args.b))
    seeds_a = run_a["data"].get("seeds")
    seeds_b = run_b["data"].get("seeds")
    print(f"[{args.label_a}] {args.a} ({seeds_a} seeds)")
    print(f"[{args.label_b}] {args.b} ({seeds_b} seeds)")
    if seeds_a != seeds_b:
        raise SystemExit("different seed counts: the pairing would be invalid")

    labels = [label for label, _, _ in PROMPTS]
    prompts = {label: prompt for label, prompt, _ in PROMPTS}
    langs = {label: lang for label, _, lang in PROMPTS}

    judges = {}
    if not args.no_judge:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        judges = {"en": Judge("gpt2", device),
                  "ru": Judge("ai-forever/rugpt3small_based_on_gpt2", device)}
        print("[judges] gpt2 + rugpt3small loaded", flush=True)

    def judge_bpb(run: dict, label: str, index: int):
        lang = langs[label]
        if lang not in judges:
            return None
        return judges[lang].bpb(prompts[label], run["samples"][label][index])

    def collect(run: dict, metric) -> list:
        out = []
        for label in labels:
            for i in range(len(run["samples"][label])):
                out.append(metric(run, label, i))
        return out

    metrics = {
        "loops": lambda run, label, i: 1.0 if loop_rate(
            [run["samples"][label][i]]) else 0.0,
        "distinct3": lambda run, label, i: distinct3([run["samples"][label][i]]),
        "code_parses": lambda run, label, i: float(code_parses(
            prompts[label], run["samples"][label][i])),
        "chars": lambda run, label, i: float(len(run["samples"][label][i])),
    }
    if judges:
        metrics["judge_bpb"] = judge_bpb

    out = {"a": args.a, "b": args.b, "label_a": args.label_a,
           "label_b": args.label_b, "seeds": seeds_a,
           "aggregate": {}, "paired": {}}
    for name in ("judge_en", "judge_ru", "self_bpb", "loops", "distinct3",
                 "code_ok"):
        va = run_a["row"].get(name)
        vb = run_b["row"].get(name)
        out["aggregate"][name] = {"a": va, "b": vb}
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            print(f"[{name:11s}] {va:.4f} -> {vb:.4f} "
                  f"({(vb - va) / va:+.1%})")

    for name, metric in metrics.items():
        va, vb = collect(run_a, metric), collect(run_b, metric)
        stats = paired(va, vb)
        out["paired"][name] = stats
        if stats["delta"] is None:
            print(f"[paired {name:11s}] not enough pairs")
            continue
        print(f"[paired {name:11s}] {args.label_b} - {args.label_a} = "
              f"{stats['delta']:+.4f} ± {stats['se']:.4f} "
              f"({stats['sigma']:.1f}σ, n={stats['n']})", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
