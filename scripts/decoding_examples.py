"""Print the actual texts behind one or more rows of the decoding sweep.

A table of judge scores can be satisfied by a model that writes two safe generic
sentences over and over, which is exactly the failure the loop column is meant to
catch.  Before choosing a recipe, read the samples.

Usage::

    python scripts/decoding_examples.py --sweep data/decoding_sweep.json \
        --pick "1.0/8/0.0/1.1" "0.8/8/0.0/1.1" --per-prompt 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def key(row: dict) -> str:
    return (f"{row['temperature']}/{row['top_k']}/"
            f"{row['top_p']}/{row['rep']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", default=["data/decoding_sweep.json"])
    ap.add_argument("--pick", nargs="+", required=True,
                    help="config keys as T/top_k/top_p/rp, e.g. 1.0/8/0.0/1.1")
    ap.add_argument("--per-prompt", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for path in args.sweep:
        rows += json.loads(Path(path).read_text(encoding="utf-8"))
    by_key = {key(r): r for r in rows}

    for want in args.pick:
        row = by_key.get(want)
        if row is None:
            print(f"[skip] {want}: not in {', '.join(args.sweep)}")
            continue
        print(f"\n### {want}  (en {row['judge_en']:.3f}, ru {row['judge_ru']:.3f}, "
              f"loops {row['loops']:.2f})")
        for label, texts in row["samples"].items():
            for text in texts[:args.per_prompt]:
                flat = text.strip().replace("\n", " ⏎ ")
                print(f"  [{label}] {flat}")


if __name__ == "__main__":
    main()
