"""Which documents did a step range consume?

With the block sampler (2026-11-01) an epoch draws every ``seq_len``-token block
at most once, so with a bin's ``train_index.npy`` -- the token span and hash of
every document that went into it -- the documents a session read follow from
three things: the step range, the batch size and the seed that built the
permutation.  Nothing needs rescanning, and nothing needs decoding.

The result is what a rebuild excludes, which is the difference between "start at
step 36,549 of a stream where that position no longer means anything" and
"start at 0 and read only what is new".
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

INDEX = "train_index.npy"


def load_index(bin_dir: str | Path) -> np.ndarray:
    """The per-document index of a bin, or a clear error if it predates it."""
    path = Path(bin_dir) / INDEX
    if not path.exists():
        raise SystemExit(
            f"{path} is missing: this bin was encoded before the index existed, "
            "so its document spans cannot be recovered without rescanning the "
            "bin and re-encoding the mix")
    return np.load(path)


def _doc_ranges(index: np.ndarray, lo_token: np.ndarray,
                hi_token: np.ndarray) -> np.ndarray:
    """Boolean mask of documents overlapped by the token intervals."""
    starts = index["start"]
    ends = index["start"] + index["n"]
    lo = np.clip(np.searchsorted(ends, lo_token, side="right"), 0, len(index) - 1)
    hi = np.searchsorted(starts, hi_token, side="right") - 1
    hi = np.clip(hi, lo, len(index) - 1)
    keep = np.zeros(len(index), dtype=bool)
    for a, b in np.unique(np.stack([lo, hi], axis=1), axis=0):
        keep[a:b + 1] = True
    return keep


def _covered_tokens(index: np.ndarray, block_start: np.ndarray,
                    seq_len: int) -> np.ndarray:
    """How many tokens of each document the drawn blocks cover.

    A block is attributed to the document holding its midpoint.  Blocks are
    drawn uniformly per *token*, so a long document is touched far more often
    than a short one -- dropping every touched document would delete several
    times more text than was actually read, which is why the caller can require
    a minimum share before a document counts as seen.
    """
    starts = index["start"]
    ends = index["start"] + index["n"]
    mid = block_start + seq_len // 2
    doc = np.clip(np.searchsorted(ends, mid, side="right"), 0, len(index) - 1)
    covered = np.zeros(len(index), dtype=np.int64)
    np.add.at(covered, doc, seq_len)
    return np.minimum(covered, index["n"].astype(np.int64))


def consumed_docs(index: np.ndarray, n_tokens: int, seq_len: int,
                  start_step: int, end_step: int, batch: int,
                  accum: int = 1, seed: int = 42, epoch: int = 0,
                  world: int = 1, rank: int = 0,
                  min_share: float = 0.0) -> np.ndarray:
    """Hashes of every document touched by steps ``[start_step, end_step)``.

    ``start_step`` is the global training step the session began at; the first
    ``start_step * batch * accum`` blocks of the epoch permutation are treated
    as already consumed, exactly as ``ByteDataset.set_shard`` does.

    ``min_share`` switches from "touched at all" (0.0) to "at least this share
    of the document's tokens was read", which is what a rebuild should exclude:
    a 200k-token document that lost 300 tokens to one block is not something to
    throw away.
    """
    n_blocks = max(1, (n_tokens - 1) // seq_len)
    steps_per_epoch = max(1, n_blocks // (batch * accum))
    skip = (start_step % steps_per_epoch) * batch * accum
    skip = min(skip, max(0, n_blocks - 1))
    n_draw = max(0, (end_step - start_step) * batch * accum)

    g = torch.Generator().manual_seed((seed + 7919 * epoch) % (2 ** 31 - 1))
    perm = torch.randperm(n_blocks, generator=g)[rank::world]
    blocks = perm[skip:skip + n_draw].numpy()
    block_start = blocks * seq_len

    if min_share <= 0:
        keep = _doc_ranges(index, block_start, block_start + seq_len)
    else:
        covered = _covered_tokens(index, block_start, seq_len)
        keep = covered >= min_share * index["n"]
    return index["h"][keep]


def save_hashes(path: str | Path, hashes: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.unique(hashes))


def merge_hashes(paths, out: str | Path) -> np.ndarray:
    parts = [np.load(p) for p in paths if Path(p).exists()]
    if not parts:
        raise SystemExit("nothing to merge")
    merged = np.unique(np.concatenate(parts))
    save_hashes(out, merged)
    return merged
