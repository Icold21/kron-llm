"""Repair an identity shard: drop every document that fails the checks and
generate replacements until the shard is whole again.

The checks live in :mod:`gen_identity` (``acceptable``), so generated and
repaired documents obey exactly the same rules: no digits, no assistant
boilerplate, no dialogue scaffolding, no framing, no misspellings.

Usage::

    python scripts/repair_identity.py --name identity_en --target 800 --style plain
    python scripts/repair_identity.py --name identity_rick_ru --target 850 --style rick
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_identity import (MANIFEST, MODEL, SHARDS, acceptable, clean,
                          generate, record_shard)

URL = "https://github.com/Icold21/kron-llm"


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}


def _near_duplicate(sig: set, kept: list[set], threshold: float = 0.8) -> bool:
    for other in kept:
        if not sig or not other:
            continue
        inter = len(sig & other)
        if inter / (len(sig) + len(other) - inter) >= threshold:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="shard stem, e.g. identity_ru or identity_rick_en")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--style", default="plain", choices=["plain", "rick"])
    ap.add_argument("--seed", type=int, default=901)
    ap.add_argument("--temperature", type=float, default=0.95)
    args = ap.parse_args()

    path = SHARDS / f"{args.name}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    lang = "ru" if args.name.endswith("_ru") else "en"

    kept, reasons = [], Counter()
    signatures: list[set] = []
    for row in rows:
        text = clean(row["text"])
        ok, why = acceptable(text, lang)
        if ok:
            sig = _shingles(text)
            if _near_duplicate(sig, signatures):
                reasons["duplicate"] += 1
                continue
            signatures.append(sig)
            row["text"] = text
            kept.append(row)
        else:
            reasons[why] += 1

    print(f"[{args.name}] {len(rows)} docs: kept {len(kept)}, "
          f"dropped {len(rows) - len(kept)} {dict(reasons)}")
    missing = max(0, args.target - len(kept))
    if missing:
        print(f"[{args.name}] generating {missing} replacements")
        fresh = generate(lang, missing, args.seed, style=args.style,
                         temperature=args.temperature)
        offset = len(kept)
        prefix = f"identity/{lang}" if args.style == "plain" else \
                 f"identity/{args.style}/{lang}"
        for i, doc in enumerate(fresh):
            doc["title"] = f"{prefix}/{offset + i:03d}"
        kept.extend(fresh)

    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept)
                    + "\n", encoding="utf-8")

    # One fresh manifest row per shard: drop the old row(s) for this path.
    if MANIFEST.exists():
        lines = [l for l in MANIFEST.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        lines = [l for l in lines
                 if Path(json.loads(l)["path"]).name != path.name]
        MANIFEST.write_text("\n".join(lines) + ("\n" if lines else ""),
                            encoding="utf-8")

    shard = record_shard(path, MANIFEST, source=f"synthetic: {MODEL}",
                         licence="CC0-1.0", url=URL, language=lang,
                         domain="identity", docs=len(kept),
                         note=f"self-identity slice, {args.style} style, "
                              "prompt-constrained; repaired against the "
                              "no-digits / no-dialogue checks")
    print(f"[done] {len(kept)} docs -> {path} ({shard.bytes/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
