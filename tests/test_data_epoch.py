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


def _n_starts(ds: ByteDataset) -> int:
    return len(ds.data) - ds.seq_len - 1


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
    n = _n_starts(ds)
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
    n = _n_starts(ds)
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
