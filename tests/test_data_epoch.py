"""Epoch sampler: full coverage, disjoint ranks, reshuffle between epochs."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kronllm.data import ByteDataset


@pytest.fixture()
def ds(tmp_path):
    path = tmp_path / "tok.bin"
    rng = np.random.default_rng(0)
    rng.integers(0, 8192, size=30_000, dtype=np.uint16).tofile(path)
    return ByteDataset(path, seq_len=64, dtype=np.uint16)


def _n_blocks(ds: ByteDataset) -> int:
    """The epoch is a permutation of blocks, not of every token offset.

    Permuting offsets (the behaviour before 2026-11-01) made one epoch cover
    63 % of the corpus and repeat the rest, and cost a 22 GB permutation for a
    2.8B-token bin.  A block is ``seq_len`` tokens, so an epoch tiles the
    corpus exactly once.
    """
    return max(1, (len(ds.data) - 1) // ds.seq_len)


def _draw_epoch(ds: ByteDataset, batch: int = 32) -> list[int]:
    """Draw batches until the epoch counter advances; return the starts seen."""
    ds._new_epoch(None)
    epoch = ds._epoch
    seen: list[int] = []
    guard = 0
    while ds._epoch == epoch and guard < 10_000:
        perm, pos = ds._perm, ds._pos
        x, _ = ds.get_batch(batch)
        if ds._epoch != epoch:          # this call wrapped into a new epoch
            break
        seen.extend(perm[pos:pos + len(x)].tolist())
        guard += 1
    return seen


def test_ranks_are_disjoint_and_cover_everything(ds):
    n = _n_blocks(ds)
    ds.set_shard(0, 2, seed=1234)
    ds._new_epoch(None)
    perm0 = ds._perm.tolist()
    ds.set_shard(1, 2, seed=1234)
    ds._new_epoch(None)
    perm1 = ds._perm.tolist()

    assert len(perm0) + len(perm1) == n
    assert abs(len(perm0) - len(perm1)) <= 1
    assert set(perm0).isdisjoint(perm1)
    assert sorted(perm0 + perm1) == list(range(n))
    assert len(set(perm0)) == len(perm0)          # no duplicates inside a rank


def test_epoch_is_a_full_pass_without_repeats(ds):
    ds.set_shard(0, 1, seed=7)
    seen = _draw_epoch(ds, batch=64)
    n = _n_blocks(ds)
    assert len(seen) == len(set(seen))            # each block once
    assert len(seen) == n - n % 64                # tail dropped, not repeated
    assert set(seen) <= set(range(n))


def test_same_seed_reproduces_and_next_epoch_reshuffles(ds):
    ds.set_shard(0, 1, seed=99)
    ds._new_epoch(None)
    first = ds._perm.tolist()
    ds.set_shard(0, 1, seed=99)
    ds._new_epoch(None)
    assert ds._perm.tolist() == first             # deterministic for a seed

    ds._new_epoch(None)                           # same dataset, next epoch
    assert ds._perm.tolist() != first


def test_batches_are_contiguous_windows(ds):
    ds.set_shard(0, 1, seed=5)
    x, y = ds.get_batch(4)
    assert x.shape == (4, ds.seq_len) and y.shape == (4, ds.seq_len)
    # y is x shifted by one token
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_resume_continues_the_epoch_instead_of_replaying_it(ds):
    """A 500-minute session must not start over from the first window.

    Sessions cover 11-35k of a 137k-step epoch, so a resumed run that restarts
    at position 0 sees exactly the windows the previous session already trained
    on and the corpus never advances.  ``skip`` is the step offset (times batch
    times accumulation) and must land on that window of the same permutation.
    """
    ds.set_shard(0, 1, seed=11)
    ds._new_epoch(None)
    perm = ds._perm.tolist()

    ds.set_shard(0, 1, seed=11, skip=25 * 8)
    x, _ = ds.get_batch(8)
    assert ds._pos == 25 * 8 + 8
    assert perm[25 * 8:25 * 8 + 8] == ds._perm[25 * 8:25 * 8 + 8].tolist()

    # Without the skip the same call returns the very first windows instead.
    ds.set_shard(0, 1, seed=11)
    ds.get_batch(8)
    assert ds._pos == 8

    # The skip is consumed once: the next epoch starts at position 0.
    ds.set_shard(0, 1, seed=11, skip=5)
    ds._new_epoch(None)
    ds._new_epoch(None)
    assert ds._pos == 0


def test_resumed_session_returns_the_same_windows(ds):
    """The windows a resumed session gets must be the ones it would have seen.

    ``test_resume_continues_the_epoch_instead_of_replaying_it`` checks
    ``_pos``; that is not enough.  Comparing ``perm[a:b]`` with the same slice
    of the same array passes even when ``get_batch`` returns the wrong windows,
    so this test compares the tensors: an uninterrupted 28-draw run must end
    with exactly the draw a session resumed at offset 27 sees first.
    """
    import torch

    ds.set_shard(0, 1, seed=11)
    tail = None
    for _ in range(28):
        tail = ds.get_batch(8)

    ds.set_shard(0, 1, seed=11, skip=27 * 8)
    x, y = ds.get_batch(8)
    assert ds._pos == 28 * 8
    assert torch.equal(x, tail[0]), "resumed window differs from the original"
    assert torch.equal(y, tail[1]), "resumed target differs from the original"

    # The target is still that window shifted by one token, so the resume lands
    # on a block boundary and the pair (x, y) stays consistent.
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_an_epoch_tiles_the_corpus_and_covers_it(ds):
    """Blocks, not token offsets: an epoch must read the corpus once.

    Permuting every token offset drew 5.4M windows out of 2.8B possible starts
    on the real corpus, so a full epoch covered only 63 % of the tokens and hit
    the rest 1.6 times.  Blocks tile the corpus, so one epoch is one honest pass.
    """
    ds.set_shard(0, 1, seed=3)
    blocks = _draw_epoch(ds, batch=32)
    n = _n_blocks(ds)

    assert len(blocks) == len(set(blocks)) == n - n % 32
    starts = [b * ds.seq_len for b in blocks]
    assert max(starts) + ds.seq_len <= len(ds.data)
    # every block is seq_len wide and no two share a token
    assert len(set(starts)) * ds.seq_len == sum(ds.seq_len for _ in starts)
    assert len(starts) * ds.seq_len >= 0.95 * len(ds.data)


