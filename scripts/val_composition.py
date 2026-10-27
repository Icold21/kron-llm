"""What is actually in the Q10 validation set, and what the eval measures.

Two facts about the training-time `val_bpb` are worth knowing before quoting it:

1. The val bin is written bucket by bucket, starting with ``en/code``, and
   ``evaluate()`` is a *sequential* pass capped at ``eval_batches`` = 64 batches
   of 8 × 512 tokens.  So the number reported every 1,000 steps is measured on
   the first 262,144 tokens of the bin, which are English Python.
2. ``bits_per_byte`` divides by the corpus-wide ``bytes_per_token`` (3.694).
   Code is ~2.3 bytes per token, so the reported bpb understates the true
   bits/byte of that span by roughly 1.6×.

This script prints the real composition of the held-out split (documents, bytes
and tokens per language and domain) so the number has a denominator.

Usage::

    python scripts/val_composition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_bilingual_bin import load_mixed, split_docs  # noqa: E402

MIX = Path("data/bilingual/mix")
OUT = Path("data/val_composition.json")
BPT = {"en": 2.917, "ru": 4.964}
VAL_FRACTION = 0.02


def main() -> None:
    docs = load_mixed(MIX)
    train, val = split_docs(docs, VAL_FRACTION)
    print(f"[val] {len(val):,} documents of {len(docs):,} "
          f"({100 * len(val) / len(docs):.2f} % of the corpus)")

    agg: dict[tuple[str, str], dict[str, int]] = {}
    for doc in val:
        key = (doc["language"], doc["domain"])
        row = agg.setdefault(key, {"docs": 0, "bytes": 0})
        row["docs"] += 1
        row["bytes"] += len(doc["text"].encode("utf-8"))

    total_bytes = sum(r["bytes"] for r in agg.values())
    total_tokens = sum(r["bytes"] / BPT[l] for (l, _), r in agg.items())
    print(f"[val] {total_bytes/1e6:.1f} MB text, ~{total_tokens/1e6:.1f}M tokens")
    print(f"{'bucket':<22}{'docs':>9}{'MB':>9}{'tokens':>11}{'share':>8}")
    for (lang, domain), row in sorted(agg.items(),
                                      key=lambda kv: -kv[1]["bytes"]):
        tokens = row["bytes"] / BPT[lang]
        print(f"{lang + '/' + domain:<22}{row['docs']:>9,}"
              f"{row['bytes']/1e6:>9.1f}{tokens/1e6:>11.2f}"
              f"{100 * tokens / total_tokens:>7.1f}%")

    lang_tokens = {l: sum(r["bytes"] / BPT[l] // 1 for (x, _), r in agg.items()
                          if x == l) for l in ("en", "ru")}
    lang_tokens = {}
    for (lang, _), row in agg.items():
        lang_tokens[lang] = lang_tokens.get(lang, 0) + row["bytes"] / BPT[lang]
    print("[val] tokens by language: "
          + ", ".join(f"{k} {100*v/total_tokens:.1f}%" for k, v in lang_tokens.items()))

    OUT.write_text(json.dumps({
        "val_fraction": VAL_FRACTION,
        "total_bytes": total_bytes,
        "est_total_tokens": total_tokens,
        "by_bucket": {f"{l}/{d}": r for (l, d), r in agg.items()},
        "tokens_by_language": lang_tokens,
    }, indent=2), encoding="utf-8")
    print(f"[val] written to {OUT}")


if __name__ == "__main__":
    main()
