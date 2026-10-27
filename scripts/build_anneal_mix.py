"""Build the curated mix for the WSD decay pass (Q10A).

The decay pass runs the last 20 % of an epoch's steps at LR 3e-4 -> 1e-6 on
data chosen for quality, not for coverage: the epoch has already taught the
model the general distribution, this pass decides what it sharpens.

Selection is deterministic (documents sorted by sha256) and budgeted per domain
by *bytes*, so the mix is reproducible and the shares are explicit.  A minority
of general text stays in on purpose: a purely encyclopedic diet narrows a small
model, and narrowing shows up as a worse bpb on the epoch-1 held-out set.

Usage::

    python scripts/build_anneal_mix.py --dry-run
    python scripts/build_anneal_mix.py
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

MIX = Path("data/bilingual/mix")
OUT = Path("data/bilingual/anneal_mix")

# Shares of the decay budget, per language and domain.  Literature carries the
# most transferable signal per token, so it takes the largest block; Russian
# literature is the single richest pool (Wikisource) and gets most of it.
# Code, logic and Cosmopedia stay as minority anchors so the model does not
# drift away from the formats it sees at inference.  The split is chosen to
# keep the epoch-1 language balance (~62 % EN / 38 % RU of tokens): budgeting
# per domain alone would have handed Russian 55 % of the tokens, because its
# literature pool is three times larger.
SHARES = {
    "ru/literature": 0.28,
    "en/literature": 0.10,
    "en/encyclopedic": 0.17,
    "ru/encyclopedic": 0.13,
    "en/science": 0.11,
    "ru/science": 0.05,
    "en/ml": 0.05,
    "en/math": 0.04,
    "ru/math": 0.02,
    "en/code": 0.03,
    "en/logic": 0.005,
    "ru/logic": 0.005,
    "ru/identity": 0.006,
    "en/identity": 0.004,
}


def domain_of(path: Path) -> tuple[str, str]:
    """``en_encyclopedic.jsonl.gz`` -> ("en", "encyclopedic")."""
    stem = path.stem.replace(".jsonl", "")
    lang, domain = stem.split("_")[:2]
    return lang, domain


def load_docs(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def take(docs: list[dict], want_bytes: int) -> tuple[list[dict], int]:
    """Deterministic sample: sort by sha256, take until the byte budget."""
    docs.sort(key=lambda d: d.get("sha256", ""))
    taken, used = [], 0
    for doc in docs:
        taken.append(doc)
        used += len(doc["text"].encode("utf-8"))
        if used >= want_bytes:
            break
    return taken, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=1660.0,
                    help="text budget in MB; 450M tokens at 3.69 bytes/token "
                         "is ~1660 MB")
    ap.add_argument("--try-mb", type=float, default=3.685,
                    help="bytes per token used to convert MB into tokens")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    budget = int(args.target_mb * 1e6)
    pools: dict[tuple[str, str], list[dict]] = {}
    sizes: dict[tuple[str, str], int] = {}
    for path in sorted(MIX.glob("*.jsonl.gz")):
        lang, domain = domain_of(path)
        pools[(lang, domain)] = load_docs(path)
        sizes[(lang, domain)] = sum(len(d["text"].encode("utf-8"))
                                    for d in pools[(lang, domain)])

    print(f"[anneal] budget {args.target_mb:.0f} MB")
    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    total_docs = 0
    lang_bytes = {"en": 0, "ru": 0}
    for domain, share in SHARES.items():
        langs = [lang for lang in ("en", "ru") if (lang, domain) in pools]
        if not langs:
            print(f"[anneal] {domain:>12}: MISSING (share {share:.0%})")
            continue
        pool_bytes = sum(sizes[(lang, domain)] for lang in langs)
        want = int(budget * share)
        for lang in langs:
            # Split the domain budget by the natural language balance of the
            # pool, so the decay pass does not shift the EN/RU ratio.
            part = int(want * sizes[(lang, domain)] / max(1, pool_bytes))
            taken, used = take(list(pools[(lang, domain)]), part)
            total_bytes += used
            total_docs += len(taken)
            lang_bytes[lang] += used
            print(f"[anneal] {domain:>12}/{lang}: {len(taken):>7,} docs, "
                  f"{used/1e6:7.1f} MB of {part/1e6:6.1f} MB "
                  f"({used/max(1, sizes[(lang, domain)]):.0%} of the pool)")
            if args.dry_run or not taken:
                continue
            out = OUT / f"{lang}_{domain}.jsonl.gz"
            with gzip.open(out, "wt", encoding="utf-8") as fh:
                for doc in taken:
                    fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    tokens = total_bytes / args.try_mb
    tot_lang = sum(lang_bytes.values()) or 1
    print(f"[anneal] total {total_docs:,} docs, {total_bytes/1e6:.0f} MB "
          f"~ {tokens/1e6:.0f}M tokens")
    print("[anneal] language by bytes: "
          + ", ".join(f"{k} {100*v/tot_lang:.1f}%" for k, v in lang_bytes.items()))
    if not args.dry_run:
        print(f"[anneal] written to {OUT}")


if __name__ == "__main__":
    main()
