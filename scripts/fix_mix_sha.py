"""Ensure every document in every mix file carries a ``sha256`` field.

``split_docs`` in ``prepare_bilingual_bin.py`` sorts by ``doc['sha256']``; the
bulk appends bypassed ``pack_shard`` and some documents arrived without it.
This rewrites the mix in place, adding the hash where it is missing.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

MIX = Path("data/bilingual/mix")


def main() -> None:
    total = fixed = 0
    for path in sorted(MIX.glob("*.jsonl.gz")):
        started = time.time()
        tmp = path.with_suffix(".tmp")
        n = added = 0
        with gzip.open(path, "rt", encoding="utf-8") as src, \
                gzip.open(tmp, "wt", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if not doc.get("sha256"):
                    doc["sha256"] = hashlib.sha256(
                        doc["text"].encode("utf-8")).hexdigest()
                    added += 1
                dst.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp, path)
        total += n
        fixed += added
        print(f"{path.name}: {n:,} docs, {added:,} hashes added, "
              f"{time.time()-started:.0f}s", flush=True)
    print(f"[done] {total:,} docs, {fixed:,} hashes added")


if __name__ == "__main__":
    main()
