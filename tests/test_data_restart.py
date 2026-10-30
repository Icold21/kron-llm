"""After the corpus is rebuilt, the run must continue, not start over.

The sampler turns a step counter into a position in a permutation of the bin.
Rebuilding the bin makes a *new* permutation over a longer stream, so there are
two ways to read "continue":

* start at position 0  -- replay the 575M tokens already trained on, which is
  exactly what we do not want;
* start at the consumed offset -- skip the 24.8 % that has been trained on and
  read the rest of the epoch once.

The second is the intent, and it is the default: ``data_step`` keeps counting
and ``skip = data_step * batch`` lands past the consumed part.  These tests pin
that down, so a future change cannot quietly turn "continue" into "restart".

The honest caveat, written here because it will otherwise be forgotten: the
permutation is new, so the offset is the same *amount* of training, not
literally the same documents.  Nothing is replayed; a random slice is delayed
to the next epoch.  Document-level "never train on this text twice" would need
a document-to-window index, which the pretraining bin does not have.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from kronllm.config import DEFAULTS, deep_merge
from kronllm.data import ByteDataset


def _dataset(tmp_path: Path, tokens: int = 200_000) -> ByteDataset:
    bin_path = tmp_path / "train.bin"
    rs = np.random.default_rng(0)
    rs.integers(0, 65535, size=tokens, dtype=np.uint16).tofile(bin_path)
    return ByteDataset(bin_path, seq_len=32, dtype=np.uint16)


def _draw(ds: ByteDataset, draws: int, batch: int = 8) -> list[torch.Tensor]:
    return [ds.get_batch(batch)[0] for _ in range(draws)]


def test_skip_lands_past_the_consumed_part(tmp_path):
    ds = _dataset(tmp_path)
    ds.set_shard(0, 1, seed=11, skip=0, epoch=0)
    every = _draw(ds, 5)

    # Resuming at step 3 with batch 8 must return draw 3, not draw 0.
    ds.set_shard(0, 1, seed=11, skip=3 * 8, epoch=0)
    resumed = _draw(ds, 2)
    assert torch.equal(resumed[0], every[3]), "resume replayed consumed data"
    assert torch.equal(resumed[1], every[4])

    # The first window of the resumed batch is not the first window overall.
    assert not torch.equal(resumed[0], every[0])


def test_position_is_consumed_steps_times_batch(tmp_path):
    ds = _dataset(tmp_path)
    ds.set_shard(0, 1, seed=11, skip=10 * 8, epoch=0)
    ds.get_batch(8)
    assert ds._pos == 11 * 8


def test_q10_continues_at_the_consumed_offset():
    """Q10 must NOT reset the stream: the point is to skip what was trained."""
    from kronllm.config import experiment_config

    cfg = experiment_config("Q10")
    assert not cfg["training"].get("data_restart_step"), (
        "Q10 must continue at the consumed offset, not restart at position 0")
    assert cfg["training"]["allow_data_change"] is True


def test_restart_knob_still_works_for_other_runs(tmp_path):
    """The knob exists for a run that genuinely wants a fresh pass."""
    ds = _dataset(tmp_path)
    ds.set_shard(0, 1, seed=11, skip=0, epoch=0)
    every = _draw(ds, 4)
    ds.set_shard(0, 1, seed=11, skip=0, epoch=0)  # what restart_step yields
    assert torch.equal(_draw(ds, 1)[0], every[0])
