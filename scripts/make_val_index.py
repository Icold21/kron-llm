"""Record where each (language, domain) bucket sits inside the val bin.

Validation should report bits per byte per slice, not one average that hides
which domain moved.  The bin is written bucket by bucket, so it only takes a
list of offsets -- but the offsets have to be exact, hence this script
re-encodes the held-out documents and **verifies the result is byte-identical**
to ``val.bpe.bin`` before writing the index.

Output: ``data/bilingual/val_index.json``

Usage::

    python scripts/make_val_index.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_bilingual_bin import load_mixed, split_docs  # noqa: E402

MIX = Path("data/bilingual/mix")
VAL_BIN = Path("data/bilingual/val.bpe.bin")
TOK = Path("data/bilingual/tokenizer.json")
OUT = Path("data/bilingual/val_index.json")
CHECK = Path("data/bilingual/_val_check.bin")
VAL_FRACTION = 0.02


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from kronllm.tokenizer import BPETokenizer

    tok = BPETokenizer(TOK, vocab_size=8192, bytes_per_token=3.694, eot_id=8191)

    docs = load_mixed(MIX)
    _, val = split_docs(docs, VAL_FRACTION)
    print(f"[index] {len(val):,} held-out documents")

    buckets: list[dict] = []
    pos = 0
    # split_docs returns train before val per bucket, so group by bucket again
    # and keep the *val bin's* order: buckets sorted, documents by sha.
    order: list[tuple[str, str]] = []
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    for doc in val:
        key = (doc["language"], doc["domain"])
        if key not in by_bucket:
            by_bucket[key] = []
            order.append(key)
        by_bucket[key].append(doc)

    with CHECK.open("wb") as fh:
        for lang, domain in order:
            items = sorted(by_bucket[(lang, domain)], key=lambda d: d["sha256"])
            start = pos
            texts = [d["text"] for d in items]
            for i in range(0, len(texts), 2000):
                for seq in tok.encode_batch(texts[i:i + 2000]):
                    arr = np.empty(len(seq) + 1, dtype=np.uint16)
                    arr[:-1] = np.asarray(seq, dtype=np.uint16)
                    arr[-1] = 8191
                    arr.tofile(fh)
                    pos += len(arr)
            buckets.append({"lang": lang, "domain": domain,
                            "docs": len(items), "start": start, "end": pos})
            print(f"[index] {lang}/{domain:<13} {len(items):>7,} docs  "
                  f"tokens {start:,} .. {pos:,}")

    same = sha256(CHECK) == sha256(VAL_BIN)
    size_same = CHECK.stat().st_size == VAL_BIN.stat().st_size
    print(f"[index] re-encoded {pos:,} tokens, identical to val.bpe.bin: {same}")
    if not (same and size_same):
        raise SystemExit("re-encoding differs from val.bpe.bin; not writing an "
                         "index that would mislabel every slice")
    CHECK.unlink()

    out = {
        "val_bin": str(VAL_BIN).replace("\\", "/"),
        "total_tokens": pos,
        "buckets": buckets,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[index] written to {OUT}")


if __name__ == "__main__":
    main()
