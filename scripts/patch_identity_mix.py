"""Replace the Rick-style identity documents in the mix after a regeneration.

The corpus build is expensive (MinHash over ~800k documents), so regenerating a
small slice does not justify a rebuild: this script rewrites the identity shard
inside `mix/<lang>_identity.jsonl.gz` and leaves everything else untouched.

Usage::

    python scripts/patch_identity_mix.py --lang ru --style rick
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
    ap.add_argument("--lang", default="ru", choices=["en", "ru"])
    ap.add_argument("--style", default="rick", choices=["rick"])
    args = ap.parse_args()

    mix_path = MIX / f"{args.lang}_identity.jsonl.gz"
    shard_path = SHARDS / f"identity_{args.style}_{args.lang}.jsonl"
    if not mix_path.exists() or not shard_path.exists():
        raise SystemExit(f"missing {mix_path} or {shard_path}")

    with gzip.open(mix_path, "rt", encoding="utf-8") as fh:
        docs = [json.loads(l) for l in fh if l.strip()]
    keep = [d for d in docs if not str(d.get("title", "")).startswith("identity/rick")]
    new = [json.loads(l) for l in shard_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    merged = keep + new
    with gzip.open(mix_path, "wt", encoding="utf-8") as fh:
        for d in merged:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[patch] {mix_path.name}: {len(docs)} -> {len(merged)} docs "
          f"({len(docs) - len(keep)} replaced)")


if __name__ == "__main__":
    main()
