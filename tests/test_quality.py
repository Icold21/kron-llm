"""Tests for the quality branch: BPE tokenizer, factorised embedding, QK-norm.

Everything here runs on CPU in a couple of seconds; the corpus/tokenizer tests
are skipped when ``data/tokenizer_meta.json`` is absent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import numpy as np

from kronllm.config import DEFAULTS, experiment_config
from kronllm.data import dataset_paths
from kronllm.model import KronTransformer, RMSNorm
from kronllm.testutils import make_cfg
from kronllm.tokenizer import (ByteTokenizer, bits_per_byte, load_tokenizer,
                               meta_path)

DATA = Path(DEFAULTS["data"]["data_dir"])
HAS_BPE = meta_path(DATA).exists()
needs_bpe = pytest.mark.skipif(not HAS_BPE, reason="run `python scripts/prepare_bpe.py`")


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
def test_bits_per_byte_math():
    # 1 nat/token with 1 byte/token == 1/ln2 bits/byte
    assert bits_per_byte(1.0, 1.0) == pytest.approx(1 / math.log(2))
    # 3 bytes/token gives three times fewer bits per byte
    assert bits_per_byte(1.0, 3.0) == pytest.approx(1 / 3 / math.log(2))
    assert bits_per_byte(0.0, 3.0) == 0.0


# ---------------------------------------------------------------------------
# byte tokenizer
# ---------------------------------------------------------------------------
def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "Once upon a time — Lily!"
    ids = tok.encode(text)
    assert len(ids) == len(text.encode("utf-8"))
    assert tok.decode(ids) == text
    assert tok.decode([65, tok.eot_id, 66]) == "AB"      # EOT is dropped
    assert tok.vocab_size == 256 and tok.eot_id == 255


def test_dataset_paths_picks_corpus():
    byte_cfg = make_cfg()
    p = dataset_paths(byte_cfg)
    assert p["train"].name == "train.bin"
    assert np.dtype(p["dtype"]).name == "uint8"

    bpe_cfg = experiment_config("Q1")
    p = dataset_paths(bpe_cfg)
    assert p["train"].name == "train.bpe.bin"
    assert np.dtype(p["dtype"]).name == "uint16"


# ---------------------------------------------------------------------------
# BPE tokenizer (needs the prepared corpus)
# ---------------------------------------------------------------------------
@needs_bpe
def test_bpe_tokenizer_roundtrip_and_budget():
    cfg = experiment_config("Q1")
    tok = load_tokenizer(cfg)
    assert tok.vocab_size == 1024 and tok.eot_id == 1023
    # ids must fit the reserved range and stay below the EOT id
    ids = tok.encode("One day, a girl named Lily found a small red ball.")
    assert ids and max(ids) < tok.eot_id
    assert tok.decode(ids) == "One day, a girl named Lily found a small red ball."


@needs_bpe
def test_bpe_compresses_about_three_bytes():
    meta = json.loads(meta_path(DATA).read_text(encoding="utf-8"))
    assert 2.5 < meta["bytes_per_token"] < 4.0, meta
    assert meta["vocab_size"] == 1024 and meta["eot_id"] == 1023


# ---------------------------------------------------------------------------
# factorised embedding
# ---------------------------------------------------------------------------
def test_factorised_embedding_params_and_shapes():
    cfg = make_cfg(**{"model.embed_rank": 32, "data.vocab_size": 512})
    model = KronTransformer(cfg)
    assert model.embed.weight.shape == (512, 32)          # V x r
    assert model.embed_up.weight.shape == (model.d, 32)   # d x r (nn.Linear(r, d))
    assert model.lm_head is None

    n_embed = sum(p.numel() for n, p in model.named_parameters() if "embed" in n)
    assert n_embed == 512 * 32 + 32 * model.d

    x = torch.randint(0, 512, (2, 7))
    logits = model(x)["logits"]
    assert logits.shape == (2, 7, 512)
    # tied head: logits == h @ U @ E^T
    h = torch.randn(3, model.d)
    ref = (h @ model.embed_up.weight) @ model.embed.weight.t()
    assert torch.allclose(model._logits(h), ref, atol=1e-5)


def test_factorised_embedding_trainable_and_tied():
    cfg = make_cfg(**{"model.embed_rank": 16, "data.vocab_size": 128})
    model = KronTransformer(cfg)
    x = torch.randint(0, 128, (2, 5))
    model(x, torch.roll(x, -1, dims=1))["loss"].backward()
    assert model.embed.weight.grad is not None
    assert model.embed_up.weight.grad is not None
    assert model.embed.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# QK-norm
# ---------------------------------------------------------------------------
def test_qk_norm_keeps_unit_rms():
    torch.manual_seed(0)
    norm = RMSNorm(24)
    q = torch.randn(4, 8, 16, 24) * 37.0
    out = norm(q)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_qk_norm_flag_builds_modules():
    plain = KronTransformer(make_cfg())
    assert not plain.blocks[0].qk_norm
    assert not hasattr(plain.blocks[0], "q_norm")

    cfg = make_cfg(**{"model.qk_norm": True})
    normed = KronTransformer(cfg)
    assert normed.blocks[0].qk_norm
    assert normed.blocks[0].q_norm.weight.numel() == normed.blocks[0].d_head
    # extra cost is 2 * d_head per layer -- negligible
    extra = normed.n_params - plain.n_params
    m = cfg["model"]
    assert extra == 2 * (m["d_model"] // m["n_heads"]) * m["n_layers"]


def test_qk_norm_forward_finite():
    cfg = make_cfg(**{"model.qk_norm": True})
    model = KronTransformer(cfg).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        out = model(x, torch.roll(x, -1, dims=1))
    assert torch.isfinite(out["logits"]).all()


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------
def test_q2_keeps_q1_activation_footprint():
    """Q2 trades parameters for width/depth at a *constant* activation footprint.

    Activation memory scales as n_layers * d_model * batch_size, so Q2 is built
    to keep exactly Q1's product while doubling the parameter count. If this
    invariant breaks, Q2 silently loses its compactness advantage.
    """
    cfg_q2 = experiment_config("Q2")
    cfg_q1 = experiment_config("Q1")
    m2 = KronTransformer(cfg_q2)
    m1 = KronTransformer(cfg_q1)

    assert m2.n_params == 2_079_992
    assert m2.n_params > 2 * m1.n_params

    # n1 * n2 must tile d_model exactly, otherwise kron_apply reshapes wrong.
    assert cfg_q2["model"]["kron"]["n1"] * cfg_q2["model"]["kron"]["n2"] == \
        cfg_q2["model"]["d_model"]
    assert cfg_q2["model"]["d_model"] % cfg_q2["model"]["n_heads"] == 0

    def act_footprint(cfg):
        return (cfg["model"]["n_layers"] * cfg["model"]["d_model"]
                * cfg["training"]["batch_size"])

    assert act_footprint(cfg_q2) == act_footprint(cfg_q1)

    fl2 = m2.flops_per_token(seq_len=cfg_q2["training"]["seq_len"],
                             batch_size=cfg_q2["training"]["batch_size"])
    fl1 = m1.flops_per_token(seq_len=cfg_q1["training"]["seq_len"],
                             batch_size=cfg_q1["training"]["batch_size"])
    assert fl2 / fl1 > 3.0, "Q2 should spend ~3x the compute per token"


def test_q1_budget_and_balance():
    cfg = experiment_config("Q1")
    model = KronTransformer(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert total <= 1_000_000, total
    assert total == model.n_params

    emb = sum(p.numel() for n, p in model.named_parameters()
              if "embed" in n or "lm_head" in n)
    mat = model.n_params_matrix
    share_emb = emb / total
    share_mat = mat / total
    assert 0.08 <= share_emb <= 0.16, f"embedding eats {share_emb:.1%} of the budget"
    assert 0.80 <= share_mat <= 0.90, f"matrices only {share_mat:.1%} of the budget"


def test_q1_uses_three_bytes_per_token():
    from kronllm.tokenizer import load_tokenizer
    if not HAS_BPE:
        pytest.skip("run `python scripts/prepare_bpe.py`")
    cfg = experiment_config("Q1")
    tok = load_tokenizer(cfg)
    # amortised training FLOPs (batch 64) -- same cost per *byte* as E2
    flops_per_byte = model_flops(cfg) / tok.bytes_per_token
    flops_per_byte_e2 = model_flops(experiment_config("E2"))
    assert 0.5 < flops_per_byte / flops_per_byte_e2 < 1.6


def model_flops(cfg):
    m = KronTransformer(cfg)
    return m.flops_per_token(seq_len=int(cfg["data"]["seq_len"]), batch_size=64)


# ---------------------------------------------------------------------------
# eval-mode prelude cache
# ---------------------------------------------------------------------------
def test_prelude_cache_is_correct_and_invalidated():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = KronTransformer(cfg).eval()
    x = torch.randint(0, 256, (2, 8))

    with torch.no_grad():
        a = model(x)["logits"]
        assert model._prelude_cache is not None     # built on the first call
        b = model(x)["logits"]                      # served from the cache
    assert torch.allclose(a, b, atol=1e-6)

    # going back to train mode must drop the cache
    model.train()
    assert model._prelude_cache is None

    # a weight change after eval() must be visible (cache cleared by eval)
    model.eval()
    with torch.no_grad():
        model.blocks[0].q.tau.add_(0.1)
        c = model(x)["logits"]
    assert not torch.allclose(a, c, atol=1e-6)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def test_csv_logger_keeps_history_across_restart(tmp_path):
    import pandas as pd

    from kronllm.train import CSVLogger

    path = tmp_path / "log.csv"
    first = CSVLogger(path)
    first.log({"step": 1, "loss": 1.0})
    first.log({"step": 2, "loss": 0.9})

    # a restarted run opens the same file: rows and schema must survive
    second = CSVLogger(path)
    second.log({"step": 3, "loss": 0.8, "val_bpb": 1.2})

    df = pd.read_csv(path)
    assert list(df["step"]) == [1, 2, 3]
    assert list(df["loss"]) == [1.0, 0.9, 0.8]
    assert "val_bpb" in df.columns

