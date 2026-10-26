"""Hard gate before the pretraining run: the corpus must meet the contract.

Usage::

    python scripts/check_corpus_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

META = Path("data/bilingual/tokenizer_meta.json")
MIN_TOKENS = 2_000_000_000
MIN_RU_SHARE = 0.30


def main() -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    tokens = int(meta["train_tokens"])
    ru = float(meta.get("token_share", {}).get("ru", 0.0))
    ok_tokens = tokens >= MIN_TOKENS
    ok_ru = ru >= MIN_RU_SHARE
    print(f"[gate] train tokens {tokens:,} (need >= {MIN_TOKENS:,}): "
          f"{'PASS' if ok_tokens else 'FAIL'}")
    print(f"[gate] Russian token share {ru*100:.1f} % "
          f"(need >= {MIN_RU_SHARE*100:.0f} %): {'PASS' if ok_ru else 'FAIL'}")
    print(f"[gate] bytes/token {meta['bytes_per_token']:.2f}, "
          f"vocab {meta['vocab_size']}")
    raise SystemExit(0 if ok_tokens and ok_ru else 1)


if __name__ == "__main__":
    main()
