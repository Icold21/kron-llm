"""Append a bulk shard to an existing mix bucket without a full rebuild.

The bulk Wikipedia import is disjoint from the curated shards by title, so it
does not need the MinHash pass (which takes hours at this document count).  The
script appends the documents to ``mix/<lang>_<domain>.jsonl.gz`` and updates the
byte totals in ``corpus_report.json`` so the published composition stays true.

Usage::

    python scripts/add_bulk_to_mix.py --shard data/bilingual/shards/wikipedia_ru_bulk.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

ROOT = Path("data/bilingual")
MIX = ROOT / "mix"
REPORT = ROOT / "corpus_report.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--lang", default="ru", choices=["en", "ru"])
    ap.add_argument("--domain", default="encyclopedic")
    args = ap.parse_args()

    shard = Path(args.shard)
    docs = [json.loads(l) for l in shard.read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    added = sum(len(d["text"].encode("utf-8")) for d in docs)

    mix_path = MIX / f"{args.lang}_{args.domain}.jsonl.gz"
    before = 0
    if mix_path.exists():
        with gzip.open(mix_path, "rt", encoding="utf-8") as fh:
            before = sum(len(json.loads(l)["text"].encode("utf-8"))
                         for l in fh if l.strip())
    with gzip.open(mix_path, "at", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    mixed = report.setdefault("mixed", {}).setdefault(args.lang, {})
    mixed[args.domain] = mixed.get(args.domain, 0) + added
    report["mixed"][args.lang] = mixed
    report["mixed_total_bytes"] = sum(sum(v.values())
                                      for v in report["mixed"].values())
    ru = sum(report["mixed"].get("ru", {}).values())
    report["ru_share"] = round(ru / max(1, report["mixed_total_bytes"]), 4)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                      encoding="utf-8")

    print(f"[mix] {mix_path.name}: {before/1e6:.0f} MB -> "
          f"{(before + added)/1e6:.0f} MB (+{added/1e6:.0f} MB, "
          f"{len(docs):,} docs)")
    print(f"[mix] total {report['mixed_total_bytes']/1e9:.2f} GB, "
          f"RU {report['ru_share']*100:.1f} % of bytes")


if __name__ == "__main__":
    main()
