"""Strip ``think``/``respond`` markers from already-fetched raw shards.

The mapper was fixed after OpenCodeReasoning had already been written, so the
existing shard still carries the tags.  Rewriting 70 MB takes a few seconds and
keeps the corpus honest: our chat template has no ``think`` marker, and a 10M
model would learn to print one.

Usage::

    python scripts/strip_think_tags.py --raw E:/kron_sft/raw --sources opencodereasoning
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

TAGS = ("<" + "think>", "<" + "/" + "think>", "<" + "respond>",
        "<" + "/" + "respond>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="E:/kron_sft/raw")
    ap.add_argument("--sources", nargs="+", default=["opencodereasoning"])
    args = ap.parse_args()
    for name in args.sources:
        path = Path(args.raw) / f"{name}.jsonl.gz"
        if not path.exists():
            print(f"[skip] {name}: no shard")
            continue
        tmp = path.with_suffix(".tmp.gz")
        changed = total = 0
        with gzip.open(path, "rt", encoding="utf-8") as src, \
                gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as dst:
            for line in src:
                row = json.loads(line)
                total += 1
                hit = False
                for message in row["messages"]:
                    text = message["content"]
                    for tag in TAGS:
                        if tag in text:
                            text = text.replace(tag, "")
                            hit = True
                    message["content"] = text
                if hit:
                    changed += 1
                    row["bytes"] = sum(len(m["content"].encode("utf-8"))
                                       for m in row["messages"])
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        print(f"[{name}] {changed:,}/{total:,} records cleaned")


if __name__ == "__main__":
    main()
