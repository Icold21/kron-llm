"""Build the corpus mix without holding every document in memory.

The old ``build_bilingual build`` loaded all 1.75 M documents (25 GB RSS on a
32 GB box) and was killed twice.  This does two passes instead:

1. stream every shard and keep only ``(sha, prefix_sha, lang, domain, bytes)``
   -- about 140 MB for 1.75 M documents -- and decide the selection there;
2. stream again and copy the selected documents into ``mix/``.

Duplicates: exact by the full-text SHA-256, near duplicates by the SHA-256 of
the first 2 KB.  The old MinHash/Jaccard check cost 18 minutes and its whole
point was to catch documents that differ only in the tail; for the same shape
of documents the 2 KB prefix catches the same pairs, and where it does not, the
difference is confined to the tail of a document that is kept whole anyway.

Quotas are per language (``QUOTAS_EN`` / ``QUOTAS_RU``): one shared table cannot
hold 60/40 in tokens, because Russian has almost no technical supply and would
stop at two thirds of its budget.

Usage::

    python scripts/build_streaming.py --budget-en-mb 4600 --budget-ru-mb 5300
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BILINGUAL = ROOT / "data" / "bilingual"
MANIFEST = BILINGUAL / "manifest.jsonl"
MIX = BILINGUAL / "mix"
REPORT = BILINGUAL / "corpus_report.json"

from kronllm.bilingual import (QUOTAS_EN, QUOTAS_RU,  # noqa: E402
                               sweep_stale_mix)


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def read_manifest() -> list[dict]:
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    seen = {}
    for r in rows:                     # last row per path wins
        seen[r["path"]] = r
    return list(seen.values())


def digest(text: str) -> tuple[str, str]:
    b = text.encode("utf-8")
    return hashlib.sha256(b).hexdigest(), hashlib.sha256(b[:2048]).hexdigest()


def text_score(text: str) -> float:
    """Cheap quality score, used to rank documents *inside* a domain.

    The fetch filters already drop the obvious junk, but the quota still has to
    pick, say, 920 MB out of 2.6 GB of English encyclopedics -- and picking by
    SHA-256 is picking at random.  This ranks by what a reader would notice:
    letters over symbols, few digits, varied vocabulary, prose-length words,
    and a document long enough to carry context.

    Deliberately cheap: no zlib, no tokenizer, only the first 4 KB, so 1.7 M
    documents cost minutes rather than an hour.
    """
    head = text[:4000]
    n = len(head) or 1
    letters = sum(1 for c in head if c.isalpha()) / n
    digits = sum(1 for c in head if c.isdigit()) / n
    words = head.split()
    if len(words) < 30:
        return 0.0
    unique = len(set(words)) / len(words)
    avg_len = sum(len(w) for w in words) / len(words)
    shape = 1.0 - min(1.0, abs(avg_len - 6.5) / 6.5)
    length_bonus = min(1.0, len(text) / 4000.0)
    return (letters * (1.0 - digits) * (0.55 + 0.45 * unique)
            * (0.5 + 0.5 * shape) * length_bonus)


def stream(shards: list[dict]):
    """Yield (shard, doc) for every document, in manifest order."""
    for shard in shards:
        path = Path(shard["path"])
        if not path.exists():
            say(f"[warn] missing {path.name}")
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield shard, json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-en-mb", type=float, default=4600)
    ap.add_argument("--budget-ru-mb", type=float, default=5300)
    args = ap.parse_args()

    budgets = {"en": int(args.budget_en_mb * 1e6), "ru": int(args.budget_ru_mb * 1e6)}
    quotas = {"en": QUOTAS_EN, "ru": QUOTAS_RU}
    shards = read_manifest()
    say(f"{len(shards)} shards, budgets EN {args.budget_en_mb:.0f} MB / "
        f"RU {args.budget_ru_mb:.0f} MB")

    # ---------------------------------------------------------------- pass 1
    seen_exact: set[str] = set()
    seen_prefix: set[str] = set()
    kept: list[tuple[str, str, str, int]] = []      # sha, lang, domain, bytes
    pool = defaultdict(int)
    per_class: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    n_docs = 0
    t0 = time.time()
    for shard, doc in stream(shards):
        n_docs += 1
        text = doc.get("text") or ""
        if len(text) < 200:
            continue
        sha, prefix = digest(text)
        if sha in seen_exact or prefix in seen_prefix:
            continue
        seen_exact.add(sha)
        seen_prefix.add(prefix)
        lang, domain = shard["language"], shard["domain"]
        size = len(text.encode("utf-8"))
        pool[(lang, domain)] += size
        per_class[(lang, domain)].append((sha, size, text_score(text)))
        if n_docs % 200_000 == 0:
            say(f"  scanned {n_docs:,} docs, kept {len(kept) + sum(len(v) for v in per_class.values()):,}, "
                f"{time.time() - t0:.0f}s")
    say(f"pass 1: {n_docs:,} documents scanned in {time.time() - t0:.0f}s, "
        f"{sum(len(v) for v in per_class.values()):,} unique")

    # ------------------------------------------------------------- selection
    selected: dict[tuple[str, str], set[str]] = {}
    report_mix: dict[str, dict[str, int]] = {}
    shortfalls = []
    total_bytes = 0
    for lang in ("en", "ru"):
        for domain, quota in quotas[lang].items():
            want = int(budgets[lang] * quota)
            have = per_class.get((lang, domain), [])
            take = 0
            chosen: set[str] = set()
            # Best first.  Ties fall back to SHA so the selection stays
            # reproducible for the same input.
            for sha, size, score in sorted(have, key=lambda x: (-x[2], x[0])):
                if take >= want:
                    break
                chosen.add(sha)
                take += size
            if not chosen:
                if want > 0:
                    shortfalls.append(f"{lang}/{domain}: nothing available")
                continue
            if take < want * 0.95:
                shortfalls.append(f"{lang}/{domain}: {take/1e6:.0f} of {want/1e6:.0f} MB")
            selected[(lang, domain)] = chosen
            report_mix.setdefault(lang, {})[domain] = take
            total_bytes += take
            say(f"  {lang}/{domain:13s} quota {want/1e6:6.0f} MB  "
                f"have {pool[(lang, domain)]/1e6:6.0f} MB  "
                f"take {take/1e6:6.0f} MB  ({len(chosen):,} docs)")

    say(f"selected {total_bytes/1e9:.2f} GB of text")
    if shortfalls:
        say("shortfalls: " + "; ".join(shortfalls))

    # ---------------------------------------------------------------- pass 2
    MIX.mkdir(parents=True, exist_ok=True)
    swept = sweep_stale_mix(MIX)
    if swept:
        say(f"  cleared {len(swept)} shards from {MIX.name}/")
    sinks = {key: gzip.open(MIX / f"{key[0]}_{key[1]}.jsonl.gz", "wt",
                            encoding="utf-8", compresslevel=6)
             for key in selected}
    written = defaultdict(int)
    docs_out = defaultdict(int)
    already: set[str] = set()          # a selected sha may appear many times
    t0 = time.time()
    try:
        for shard, doc in stream(shards):
            key = (shard["language"], shard["domain"])
            chosen = selected.get(key)
            if not chosen:
                continue
            text = doc.get("text") or ""
            if len(text) < 200:
                continue
            sha, _ = digest(text)
            if sha not in chosen or sha in already:
                continue
            already.add(sha)
            sinks[key].write(json.dumps(doc, ensure_ascii=False) + "\n")
            written[key] += len(text.encode("utf-8"))
            docs_out[key] += 1
    finally:
        for fh in sinks.values():
            fh.close()
    say(f"pass 2: wrote {sum(docs_out.values()):,} documents in {time.time() - t0:.0f}s")

    en_bytes = sum(v for k, v in written.items() if k[0] == "en")
    ru_bytes = sum(v for k, v in written.items() if k[0] == "ru")
    total = en_bytes + ru_bytes
    mixed = {l: {} for l in ("en", "ru")}
    for (lang, domain), size in list(written.items()):
        mixed[lang][domain] = int(size)
    REPORT.write_text(json.dumps({
        "mixed": mixed,
        "mixed_total_bytes": int(total),
        "ru_share": round(ru_bytes / max(1, total), 4),
        "documents": {l: sum(v for k, v in docs_out.items() if k[0] == l)
                      for l in ("en", "ru")},
        "shortfalls": shortfalls,
        "budgets_mb": {"en": args.budget_en_mb, "ru": args.budget_ru_mb},
        "dedupe": "exact sha256 + 2 KB prefix",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"DONE: EN {en_bytes/1e9:.2f} GB, RU {ru_bytes/1e9:.2f} GB, "
        f"total {total/1e9:.2f} GB, RU share {ru_bytes/total:.1%}")


if __name__ == "__main__":
    main()
