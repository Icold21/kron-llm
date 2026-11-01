"""Recover ``train_index.npy`` for a bin that was encoded before the index existed.

The bin is EOT-delimited, so the document spans are the positions of id
``eot_id``: document *i* starts after the previous EOT and ends at its own.  The
span and its hash are all the seen-set needs, and the hash is of tokens, so
nothing has to be decoded or re-encoded.

Usage::

    python scripts/build_bin_index.py --bin E:/kron_bins/epoch_v4/train.bpe.bin \\
        --eot 8191 --out E:/kron_bins/epoch_v4/train_index.npy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

CHUNK = 1 << 26          # 67M tokens per read: ~128 MB, keeps the scan cache-friendly


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--eot", type=int, default=8191)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hash", action="store_true", default=True,
                    help="also hash every document (default: on)")
    args = ap.parse_args()

    t0 = time.time()
    path = Path(args.bin)
    out = Path(args.out) if args.out else path.with_name("train_index.npy")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(data)

    eot = []
    for start in range(0, n, CHUNK):
        chunk = np.asarray(data[start:start + CHUNK])
        eot.append(np.flatnonzero(chunk == args.eot) + start)
    eot = np.concatenate(eot)
    print(f"{len(eot):,} documents in {n:,} tokens ({time.time()-t0:.0f}s)")

    starts = np.concatenate([[0], eot[:-1] + 1]).astype(np.int64)
    ends = eot.astype(np.int64) + 1
    counts = (ends - starts).astype(np.int64)

    from kronllm.tokenizer import doc_hash   # noqa: E402  (needs the repo path)

    hashes = np.empty(len(eot), dtype=np.uint64)
    for i, (s, c) in enumerate(zip(starts, counts)):
        hashes[i] = doc_hash(data[s:s + c])
        if (i + 1) % 100_000 == 0:
            print(f"  hashed {i+1:,} / {len(eot):,} ({time.time()-t0:.0f}s)")

    index = np.zeros(len(eot), dtype=[("start", "<i8"), ("n", "<i4"),
                                     ("h", "<u8")])
    index["start"] = starts
    index["n"] = counts
    index["h"] = hashes
    np.save(out, index)
    print(f"{out} -> {len(index):,} documents, "
          f"{int(index['n'].sum()):,} tokens, {time.time()-t0:.0f}s")
    print(f"unique hashes {len(np.unique(hashes)):,}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
