"""Repeat the identity documents inside the mix so the slice is not starved.

The epoch sampler shuffles 512-token windows across the whole stream, so the
identity block is already spread over the epoch rather than seen as one chunk.
The real problem is the *share*: ~0.02 % of the tokens.  A 10M model needs to
see the name and the facts more often than once per 5M tokens.

This appends ``factor - 1`` extra copies of every identity document to the mix
(no corpus rebuild, no dedup pass).  The copies are exact, which is what the
slice is for: the facts are few and the phrasing is already diverse.

Usage::

    python scripts/upsample_identity_mix.py --factor 5
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

MIX = Path("data/bilingual/mix")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=5)
    args = ap.parse_args()

    for lang in ("en", "ru"):
        path = MIX / f"{lang}_identity.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            docs = [json.loads(line) for line in fh if line.strip()]
        before = sum(len(d["text"].encode()) for d in docs)
        with gzip.open(path, "at", encoding="utf-8") as fh:
            for _ in range(args.factor - 1):
                for doc in docs:
                    fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        after = before * args.factor
        print(f"[identity] {lang}: {len(docs)} docs, "
              f"{before/1e6:.2f} MB -> {after/1e6:.2f} MB "
              f"(x{args.factor})")


if __name__ == "__main__":
    main()
