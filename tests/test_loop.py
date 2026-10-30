"""Tests for the looped (weight-tied) execution schedule.

The plan is a small, sharp thing to get wrong: it decides which physical block
each execution slot runs, how many KV slots exist, and when the per-pass bias
fires.  These tests pin the three properties the rest of the code relies on:
the default is the old modulo scheme, a middle band leaves the head and tail
unshared, and neither changes the parameter count.
"""
from __future__ import annotations

import torch

from kronllm.config import experiment_config
from kronllm.model import KronTransformer
from kronllm.testutils import make_cfg


def build(**overrides):
    return KronTransformer(make_cfg(**overrides))


def test_default_plan_is_the_modulo_scheme():
    """recurse with no band repeats the whole stack, exactly as before."""
    m = build(**{"model.n_unique_blocks": 6, "model.recurse": 2})
    assert m.plan == tuple(list(range(6)) * 2), m.plan
    assert m.pass_index == (0,) * 6 + (1,) * 6, m.pass_index
    assert m.n_layers == 12


def test_middle_band_keeps_head_and_tail_unshared():
    m = build(**{"model.n_unique_blocks": 12, "model.recurse": 2,
                 "model.loop_start": 2, "model.loop_end": 10})
    # 2 head + 8 looped twice + 2 tail = 20 executions from 12 blocks.
    assert m.plan == (0, 1) + tuple(range(2, 10)) * 2 + (10, 11), m.plan
    assert m.n_layers == 20
    # Tail layers run after the loop finished, so they belong to the last pass.
    assert m.pass_index[:2] == (0, 0)
    assert set(m.pass_index[2:18]) == {0, 1}
    assert m.pass_index[18:] == (1, 1)
    # No new parameters: the same blocks, referenced twice.
    assert m.n_params == build(**{"model.n_unique_blocks": 12}).n_params


def test_rejects_a_loop_outside_the_stack():
    for start, end in ((5, 5), (6, 3), (-1, 4), (0, 99)):
        try:
            build(**{"model.n_unique_blocks": 6, "model.recurse": 2,
                     "model.loop_start": start, "model.loop_end": end})
        except ValueError:
            continue
        raise AssertionError(f"accepted loop_start={start} loop_end={end}")


def test_forward_backprops_through_a_shared_visit():
    m = build(**{"model.n_unique_blocks": 6, "model.recurse": 2,
                 "model.loop_start": 1, "model.loop_end": 5,
                 "model.pass_embedding": True})
    ids = torch.randint(0, 256, (2, 16))
    assert m(ids)["logits"].shape[:2] == (2, 16)
    m(ids, torch.roll(ids, -1, dims=1))["loss"].backward()
    assert m.blocks[2].q.A.grad is not None, "a looped block saw no gradient"
    assert torch.isfinite(m.blocks[2].q.A.grad).all()
    # One execution slot per visit, not one per physical block: gradients from
    # both visits accumulate into the same parameter.
    assert len(m.blocks) == m.n_layers == 10
    assert m.plan == (0,) + (1, 2, 3, 4) * 2 + (5,), m.plan


def test_set_recurse_rebuilds_the_band_not_the_whole_stack():
    m = build(**{"model.n_unique_blocks": 6, "model.recurse": 1,
                 "model.loop_start": 2, "model.loop_end": 5})
    assert m.n_layers == 6
    m.set_recurse(3)
    assert m.plan == (0, 1) + (2, 3, 4) * 3 + (5,), m.plan
    assert m.n_layers == 12
    assert m.blocks[2] is m.blocks[5] is m.blocks[8]


def test_pass_bias_fires_once_per_pass():
    m = build(**{"model.n_unique_blocks": 6, "model.recurse": 2,
                 "model.loop_start": 2, "model.loop_end": 5,
                 "model.pass_embedding": True})
    transitions = [i for i in range(len(m.pass_index))
                   if m.pass_index[i] != (m.pass_index[i - 1] if i else -1)]
    # Start of pass 0 (slot 0) and start of pass 1 (slot 2 + 3 = 5).
    assert transitions == [0, 5], transitions


def test_q10_is_untouched():
    """The live run must not move: a config change here would restart it."""
    m = KronTransformer(experiment_config("Q10"))
    assert m.recurse == 1 and m.n_layers == 25
    assert m.pass_bias is None
    assert m.plan == tuple(range(25))
