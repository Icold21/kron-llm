"""Rebuild the identity slice inside the mix after a regeneration.

The corpus build is expensive (MinHash over ~800k documents), so regenerating a
small slice does not justify a rebuild: this script rewrites
`mix/<lang>_identity.jsonl.gz` from the identity shards of that language and
leaves everything else untouched.

Usage::

    python scripts/patch_identity_mix.py --lang ru
    python scripts/patch_identity_mix.py --lang en --lang ru
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

ROOT = Path("data/bilingual")
SHARDS = ROOT / "shards"
MIX = ROOT / "mix"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", action="append", choices=["en", "ru"],
                    help="language to patch; repeat for both")
    args = ap.parse_args()
    langs = args.lang or ["en", "ru"]

    for lang in langs:
        mix_path = MIX / f"{lang}_identity.jsonl.gz"
        sources = [SHARDS / f"identity_{lang}.jsonl",
                   SHARDS / f"identity_rick_{lang}.jsonl"]
        missing = [p for p in sources if not p.exists()]
        if missing:
            raise SystemExit(f"missing {missing}")

        docs = []
        for path in sources:
            docs.extend(json.loads(line) for line in
                        path.read_text(encoding="utf-8").splitlines()
                        if line.strip())

        before = 0
        if mix_path.exists():
            with gzip.open(mix_path, "rt", encoding="utf-8") as fh:
                before = sum(1 for line in fh if line.strip())

        with gzip.open(mix_path, "wt", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"[patch] {mix_path.name}: {before} -> {len(docs)} docs")


if __name__ == "__main__":
    main()
