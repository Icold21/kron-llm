"""Re-tokenise the existing byte corpus into a byte-level BPE memmap.

Reads ``data/train.bin`` / ``data/val.bin`` (0xFF-separated UTF-8 stories), so
the BPE model sees *exactly* the same text as the byte baseline -- that is what
makes bits-per-byte comparable across tokenizers.

Usage
-----
    python scripts/prepare_bpe.py                       # vocab 1024 (~3 bytes/token)
    python scripts/prepare_bpe.py --vocab 2048
    python scripts/prepare_bpe.py --vocab 512 --merge-stories 40000
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/prepare_bpe.py` from a fresh checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from pathlib import Path

from kronllm.tokenizer import (BPETokenizer, encode_corpus, meta_path,
                               stories_from_bytes, train_bpe_merges)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--vocab", type=int, default=1024,
                    help="total model vocabulary, including the EOT id")
    ap.add_argument("--merge-stories", type=int, default=60000,
                    help="stories used to learn the merges")
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    t0 = time.time()
    vocab = int(args.vocab)
    eot_id = vocab - 1
    tok_path = data_dir / "tokenizer.json"

    train_stories = stories_from_bytes(data_dir / "train.bin")
    val_stories = stories_from_bytes(data_dir / "val.bin")
    print(f"[bpe] train stories {len(train_stories):,}, "
          f"val stories {len(val_stories):,}")

    sample = train_stories[:args.merge_stories]
    print(f"[bpe] learning merges on {len(sample):,} stories "
          f"({sum(len(s) for s in sample) / 1e6:.1f} MB) ...")
    train_bpe_merges([s.decode("utf-8", errors="replace") for s in sample],
                     vocab_size=vocab - 1, out_path=tok_path,
                     min_frequency=args.min_frequency)

    tok = BPETokenizer(tok_path, vocab_size=vocab, bytes_per_token=1.0,
                       eot_id=eot_id)
    print(f"[bpe] learned {tok._tok.get_vocab_size()} ids "
          f"(+ EOT -> {vocab})")

    tr = encode_corpus(train_stories, tok, data_dir / "train.bpe.bin", eot_id)
    va = encode_corpus(val_stories, tok, data_dir / "val.bpe.bin", eot_id)
    bpt = va["bytes"] / va["tokens"]
    print(f"[bpe] train {tr['tokens']:,} tokens "
          f"({tr['bytes_per_token']:.2f} bytes/token)")
    print(f"[bpe] val   {va['tokens']:,} tokens "
          f"({va['bytes_per_token']:.2f} bytes/token)")

    meta = {
        "type": "bpe",
        "tokenizer": str(tok_path).replace("\\", "/"),
        "vocab_size": vocab,
        "eot_id": eot_id,
        "bytes_per_token": bpt,
        "train_tokens": tr["tokens"],
        "val_tokens": va["tokens"],
        "train_bytes": tr["bytes"],
        "val_bytes": va["bytes"],
        "merge_stories": len(sample),
    }
    meta_path(data_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[bpe] wrote {meta_path(data_dir)} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
