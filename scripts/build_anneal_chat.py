"""Build the chat-format slice that goes into the decay pass (Q10A).

The four chat tokens in the vocabulary -- ``<|system|>``, ``<|user|>``,
``<|assistant|>``, ``<|eot|>`` -- are **cold** at the end of pretraining: a scan
of ``train.bpe.bin`` (2.25B tokens) finds 0, 5, 6 and 0 occurrences of them.
The model has never seen a dialogue, so the first few hundred SFT steps would be
spent learning four embedding rows and a markup instead of learning to answer.

The decay pass is the cheapest place to fix that: it is the last 30 % of an
epoch at LR 3e-4 -> 1e-6, and 2.5 % of it (about 16.6M tokens) is enough for the
format to stop being new.  The records here are rendered with the *same*
``build_sft.render`` that the SFT corpus uses, so the tokens the model meets in
the decay pass are exactly the tokens it will meet in SFT.

The pools land in ``data/bilingual/mix`` next to the other domains and are
consumed by ``build_anneal_mix.py``; the share of each is set there.

Usage::

    python scripts/build_anneal_chat.py
    python scripts/build_anneal_chat.py --target-mb 80
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_sft import render  # noqa: E402

RAW = Path(r"E:\kron_sft\raw")
MIX = ROOT / "data" / "bilingual" / "mix"

# (shard, language, source label, licence).  English is real multi-turn
# dialogue; Russian has almost none on the Hub under a usable licence, so its
# slice leans on scored dialogues plus instruction pairs rendered in the same
# markup -- the point of this pool is the *format*, not the register.
SOURCES: List[Tuple[str, str, str, str]] = [
    ("ultrachat", "en", "UltraChat 200k", "MIT"),
    ("slimorca", "en", "SlimOrca", "MIT"),
    ("oasst_en", "en", "OpenAssistant oasst1", "Apache-2.0"),
    ("saiga_scored", "ru", "Saiga scored", "Apache-2.0"),
    ("oasst_ru", "ru", "OpenAssistant oasst1", "Apache-2.0"),
    ("russian_instructions2", "ru", "Russian Instructions 2", "MIT"),
]

MIN_CHARS = 80
MAX_CHARS = 4000


def records(path: Path) -> List[dict]:
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=80.0,
                    help="per-language pool size; the decay mix takes ~2.5 %% "
                         "of the pass, so this is ~2.5x the bytes needed")
    ap.add_argument("--out-dir", default=str(MIX))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = int(args.target_mb * 1e6)

    pools: Dict[str, list] = {"en": [], "ru": []}
    seen: Dict[str, set] = {"en": set(), "ru": set()}
    used: Dict[str, int] = {"en": 0, "ru": 0}

    for shard, lang, label, licence in SOURCES:
        if used[lang] >= target:
            print(f"  [full] {lang} pool is at {used[lang] / 1e6:.1f} MB")
            continue
        kept = 0
        for rec in records(RAW / f"{shard}.jsonl.gz"):
            if used[lang] >= target:
                break
            messages = rec.get("messages") or []
            if len(messages) < 2:
                continue
            text, _ = render(messages)
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen[lang]:
                continue
            seen[lang].add(digest)
            pools[lang].append({
                "title": "",
                "text": text,
                "language": lang,
                "domain": "chat",
                "source": label,
                "licence": licence,
                "url": "",
                "sha256": digest,
            })
            used[lang] += len(text.encode("utf-8"))
            kept += 1
        print(f"  [{shard}] +{kept:,} records, {lang} pool "
              f"{used[lang] / 1e6:.1f} MB")

    for lang, docs in pools.items():
        path = out_dir / f"{lang}_chat.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
            for doc in docs:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"[pool] {path} {len(docs):,} records, {used[lang] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
