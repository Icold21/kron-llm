"""Record which documents a session read, so the next rebuild skips only those.

Usage::

    # after a session: what steps 36,549 -> 52,000 consumed
    python scripts/seen_from_index.py --bin-dir E:/kron_bins/epoch_v4 \\
        --start-step 36549 --end-step 52000 --out data/seen/epoch_v4_52000.npy

    # merge everything a run has read and report the share of the corpus
    python scripts/seen_from_index.py --merge data/seen/*.npy \\
        --bin-dir E:/kron_bins/epoch_v4 --out data/seen/all.npy

The hashes are of *tokens*, so the same document recognised in the next bin
without touching the text; ``prepare_bilingual_bin.py --skip-seen`` drops them
at encode time and ``--target-tokens`` stops the over-drawn mix at the intended
size, which is what makes the start step exact instead of positional.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kronllm.seen import (consumed_docs, load_index, merge_hashes,  # noqa: E402
                          save_hashes)

META = "tokenizer_meta.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="E:/kron_bins/epoch_v4")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--end-step", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="a document counts as seen once this share of its "
                         "tokens was read (0 = any touch; blocks are drawn per "
                         "token, so 'touched' over-selects long documents)")
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bin_dir = Path(args.bin_dir)
    index = load_index(bin_dir)
    meta = json.loads((bin_dir / META).read_text(encoding="utf-8"))
    n_tokens = int(meta["train_tokens"])

    if args.merge:
        hashes = merge_hashes(args.merge, args.out)
        print(f"merged {len(args.merge)} sets -> {len(hashes):,} documents")
    else:
        if not args.end_step:
            raise SystemExit("--end-step is required (or use --merge)")
        hashes = consumed_docs(index, n_tokens, args.seq_len, args.start_step,
                               args.end_step, args.batch, args.accum,
                               args.seed, args.epoch, min_share=args.min_share)
        save_hashes(args.out, hashes)
        print(f"steps {args.start_step:,} -> {args.end_step:,} "
              f"({args.end_step - args.start_step:,} steps = "
              f"{(args.end_step - args.start_step) * args.batch * args.accum * args.seq_len:,} "
              f"tokens drawn): {len(hashes):,} of {len(index):,} documents "
              f"are at least {args.min_share:.0%} read")

    known = np.isin(index["h"], hashes)
    print(f"{100 * len(hashes) / len(index):.1f} % of the documents, "
          f"{int(index['n'][known].sum()):,} of {n_tokens:,} tokens "
          f"({100 * int(index['n'][known].sum()) / n_tokens:.1f} %)")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
