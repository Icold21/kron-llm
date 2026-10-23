"""Tests for the routed dense embedding (Q3).

The routed embedding replaces the factorised lookup with a top-k mixture of a
dense bank.  These tests cover the parameter budget, the mixture math, gradient
flow, the adjustable top-k and the tied effective head.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from kronllm.config import experiment_config
from kronllm.model import KronTransformer
from kronllm.testutils import make_cfg


def _small_routed() -> KronTransformer:
    cfg = make_cfg(**{
        "model.embed_mode": "routed",
        "model.embed_rank": 0,
        "model.routed.v_eff": 64,
        "model.routed.route_dim": 4,
        "model.routed.top_k": 5,
    })
    return KronTransformer(cfg)


def test_routed_embedding_mixture_math():
    model = _small_routed()
    routed = model.routed
    logits = routed.route_logits()
    values, indices = logits.topk(routed.top_k, dim=-1)
    weights = F.softmax(values, dim=-1)
    manual = (routed.bank(indices) * weights.unsqueeze(-1)).sum(dim=-2)
    eff = routed.effective_embeddings()
    assert eff.shape == (model.vocab_size, model.d)
    assert torch.allclose(eff, manual, atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(model.vocab_size),
                          atol=1e-6)


def test_routed_embedding_gradients():
    model = _small_routed()
    idx = torch.randint(0, model.vocab_size, (2, 6))
    targets = torch.randint(0, model.vocab_size, (2, 6))
    out = model(idx, targets)
    out["loss"].backward()
    assert model.routed.bank.weight.grad is not None
    assert model.routed.route.weight.grad is not None
    assert model.routed.route_proj.weight.grad is not None
    assert model.routed.bank.weight.grad.abs().sum() > 0
    assert model.routed.route_proj.weight.grad.abs().sum() > 0


def test_routed_embedding_tied_head_and_lookup():
    model = _small_routed().eval()
    idx = torch.randint(0, model.vocab_size, (2, 6))
    eff = model._effective_embeddings()
    assert torch.allclose(model._embed_tokens(idx), F.embedding(idx, eff))
    x = torch.randn(2, 6, model.d)
    assert torch.allclose(model._logits(x, eff), x @ eff.t(), atol=1e-6)


def test_routed_top_k_is_adjustable_and_invalidates_cache():
    model = _small_routed().eval()
    idx = torch.randint(0, model.vocab_size, (1, 4))
    before = model(idx)["logits"]
    assert model._eff_cache is not None
    model.set_route_top_k(3)
    after = model(idx)["logits"]
    assert model.routed.top_k == 3
    assert model._eff_cache is not None
    assert not torch.allclose(before, after)
    model.train()
    assert model._eff_cache is None


def test_q3_parameter_budget():
    model = KronTransformer(experiment_config("Q3"))
    assert model.n_params == 3_544_104
    assert model.n_params_matrix == 1_890_360
    assert model.embed_mode == "routed"
