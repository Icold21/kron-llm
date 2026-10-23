"""Unit tests T1-T10 from the technical specification (section 6.1)."""
from __future__ import annotations

import math

import pytest
import torch

from kronllm.kron import (KronCompose, apply_rope, effective_rank, kron_apply,
                          kron_materialize, rope_frequencies)


# ---------------------------------------------------------------------------
# T1 -- kron_apply == (A (x) B) x == A X B^T
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n1,n2", [(8, 16), (4, 4), (12, 16)])
def test_T1_kron_apply_matches_dense(n1, n2):
    torch.manual_seed(0)
    A = torch.randn(n1, n1, dtype=torch.float64)
    B = torch.randn(n2, n2, dtype=torch.float64)
    X = torch.randn(n1, n2, dtype=torch.float64)

    y = kron_apply(A, B, X)
    W = kron_materialize(A, B)
    y_dense = (W @ X.reshape(-1)).reshape(n1, n2)
    y_manual = A @ X @ B.T

    assert torch.allclose(y, y_manual, atol=1e-10)
    assert torch.allclose(y, y_dense, atol=1e-10)


def test_T1_batched():
    torch.manual_seed(0)
    A, B = torch.randn(8, 8), torch.randn(16, 16)
    x = torch.randn(3, 5, 8, 16)
    ref = torch.stack([torch.stack([A @ x[b, t] @ B.T for t in range(5)])
                       for b in range(3)])
    assert torch.allclose(kron_apply(A, B, x), ref, atol=1e-5)


# ---------------------------------------------------------------------------
# T2 -- with zero innovation, KronCompose == s^L * I
# ---------------------------------------------------------------------------
def test_T2_identity_when_D_is_zero():
    m = KronCompose(64, 64, 8, 8, L=4, K=2, K_win=2, s_init=0.5,
                    anchor=True, rezero_eps=0.0)
    with torch.no_grad():
        m.A.zero_()
        m.B.zero_()
    x = torch.randn(2, 7, 64)
    y, D = m(x, None, layer_idx=0)
    assert torch.allclose(D, torch.zeros_like(D))
    # y = gamma * s^L * x
    expected = float(m.s) ** m.L * x * float(m.gate)
    assert torch.allclose(y, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# T3 -- L=1, K=1 -> (sI + A (x) B) x
# ---------------------------------------------------------------------------
def test_T3_single_factor():
    m = KronCompose(64, 64, 8, 8, L=1, K=1, K_win=0, s_init=0.3,
                    anchor=True, rezero_eps=0.0,
                    normalize_D=False, innovation_scale=1.0)
    x = torch.randn(2, 5, 64)
    y, D = m(x, None, layer_idx=1)
    W = m.A[0, 0], m.B[0, 0]
    dense = m.materialize(D)[0]
    assert torch.allclose(dense, float(m.s) * torch.eye(64) + kron_materialize(*W),
                          atol=1e-5)
    expected = float(m.s) * torch.eye(64) + kron_materialize(*W)
    out = (expected @ x[0].T).T * float(m.gate)
    assert torch.allclose(y[0], out, atol=1e-5)


# ---------------------------------------------------------------------------
# T3b -- the fused "matrix" fast path must equal the sequential composition
# ---------------------------------------------------------------------------
def test_T3b_matrix_mode_matches_sequential():
    torch.manual_seed(0)
    m = KronCompose(64, 64, 8, 8, L=4, K=2, K_win=2, s_init=0.5,
                    anchor=True, rezero_eps=0.01).double()
    x = torch.randn(2, 300, 64, dtype=torch.float64)

    y_s, D = m(x, None, 0, mode="sequential")
    y_m, Dm = m(x, None, 0, mode="matrix")
    assert torch.allclose(D, Dm, atol=1e-12)
    assert torch.allclose(y_s, y_m, atol=1e-12), (y_s - y_m).abs().max().item()

    # ... and with a non-trivial incoming depth state
    y_s, _ = m(x[:1], D, 1, mode="sequential")
    y_m, D1 = m(x[:1], D, 1, mode="matrix")
    assert torch.allclose(y_s, y_m, atol=1e-12)

    # ... and the fused product equals the explicit dense product
    W = m.materialize(D1)
    P = W[3] @ W[2] @ W[1] @ W[0]
    ref = (x[:1].reshape(-1, 64) @ P.T).reshape(1, 300, 64) * float(m.gate)
    assert torch.allclose(y_m, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# T4 -- recurrence D_i = I_i + sigma(theta) * D_{i-1}
# ---------------------------------------------------------------------------
def test_T4_depth_recurrence():
    # raw (un-normalised) recurrence: D_i = I_i + sigma(theta) * D_{i-1}
    m = KronCompose(64, 64, 8, 8, L=3, K=2, K_win=0, s_init=0.1,
                    anchor=True, rezero_eps=0.0, stop_gradient=True,
                    normalize_D=False, innovation_scale=1.0)
    x = torch.randn(1, 4, 64)
    _, D0 = m(x, None, layer_idx=0)
    _, D1 = m(x, D0, layer_idx=1)

    I = m.innovation().detach()
    sig = torch.sigmoid(m.theta).detach().view(-1, 1, 1)
    assert torch.allclose(D0, I, atol=1e-6)
    assert torch.allclose(D1, I + sig * D0, atol=1e-6)


def test_T4_normalised_recurrence_has_fixed_rms():
    """With normalize_D every channel of D has RMS == tau, at any depth."""
    m = KronCompose(64, 64, 8, 8, L=3, K=2, K_win=0, s_init=1.0,
                    anchor=True, rezero_eps=0.0, stop_gradient=True,
                    normalize_D=True, innovation_scale=0.25)
    x = torch.randn(1, 4, 64)
    _, D0 = m(x, None, layer_idx=0)
    _, D1 = m(x, D0, layer_idx=1)
    _, D2 = m(x, D1, layer_idx=2)
    for D in (D0, D1, D2):
        rms = D.pow(2).mean(dim=(-2, -1)).sqrt()
        assert torch.allclose(rms, torch.full_like(rms, 0.25), atol=1e-5)


def test_T4_window_resets():
    m = KronCompose(64, 64, 8, 8, L=2, K=2, K_win=2, s_init=0.1,
                    anchor=True, rezero_eps=0.0,
                    normalize_D=False, innovation_scale=1.0)
    x = torch.randn(1, 3, 64)
    _, D0 = m(x, None, layer_idx=0)
    _, D1 = m(x, D0, layer_idx=1)         # same window -> accumulates
    _, D2 = m(x, D1, layer_idx=2)         # window boundary -> reset
    I = m.innovation().detach()
    assert torch.allclose(D2, I, atol=1e-6)
    assert not torch.allclose(D1, I, atol=1e-6)


# ---------------------------------------------------------------------------
# T5 -- stop-gradient on D_{i-1}
# ---------------------------------------------------------------------------
def test_T5_stop_gradient():
    m = KronCompose(64, 64, 8, 8, L=2, K=2, K_win=0, stop_gradient=True)
    x = torch.randn(1, 3, 64, requires_grad=True)
    prev = torch.randn(2, 64, 64, requires_grad=True)
    y, D = m(x, prev, layer_idx=1)
    y.sum().backward()
    assert prev.grad is None, "D_{i-1} must be detached"
    assert x.grad is not None


def test_T5_full_backprop_optional():
    m = KronCompose(64, 64, 8, 8, L=2, K=2, K_win=0, stop_gradient=False)
    x = torch.randn(1, 3, 64, requires_grad=True)
    prev = torch.randn(2, 64, 64, requires_grad=True)
    y, _ = m(x, prev, layer_idx=1)
    y.sum().backward()
    assert prev.grad is not None


# ---------------------------------------------------------------------------
# T6 -- RoPE preserves the norm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dh", [16, 8, 32])
def test_T6_rope_norm(dh):
    torch.manual_seed(0)
    T = 17
    cos, sin = rope_frequencies(T, dh, 10000.0)
    cos, sin = cos.view(1, 1, T, -1), sin.view(1, 1, T, -1)
    x = torch.randn(2, 3, T, dh)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_T6_rope_is_position_dependent():
    cos, sin = rope_frequencies(8, 16, 10000.0)
    cos, sin = cos.view(1, 1, 8, -1), sin.view(1, 1, 8, -1)
    x = torch.ones(1, 1, 8, 16)
    y = apply_rope(x, cos, sin)
    assert not torch.allclose(y[:, :, 0], y[:, :, 1])


# ---------------------------------------------------------------------------
# T7 -- causal mask
# ---------------------------------------------------------------------------
def test_T7_attention_is_causal(model):
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        base = model(x)["logits"]
        x2 = x.clone()
        x2[0, 12] = (x2[0, 12] + 7) % 256
        alt = model(x2)["logits"]
    # positions strictly before the modified token must be unaffected
    assert torch.allclose(base[0, :12], alt[0, :12], atol=1e-5)
    # and the modified position itself must change
    assert not torch.allclose(base[0, 12:], alt[0, 12:], atol=1e-6)


def test_T7_manual_mask():
    T = 6
    scores = torch.zeros(T, T)
    mask = torch.ones(T, T, dtype=torch.bool).triu(1)
    masked = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(masked, dim=-1)
    for i in range(T):
        for j in range(T):
            if j > i:
                assert p[i, j] == 0.0


# ---------------------------------------------------------------------------
# T8 -- weight tying
# ---------------------------------------------------------------------------
def test_T8_weight_tying(model):
    assert model.lm_head.weight is model.embed.weight
    assert model.embed.weight.shape == (256, model.d)


# ---------------------------------------------------------------------------
# T9 -- parameter budget
# ---------------------------------------------------------------------------
def test_T9_param_budget():
    from kronllm.config import load_config
    from kronllm.model import KronTransformer
    cfg = load_config()
    m = KronTransformer(cfg)
    assert m.n_params <= 1_000_000, f"{m.n_params:,} params > 1M"
    # the spec targets ~190K for E2
    assert 150_000 <= m.n_params <= 230_000, f"{m.n_params:,} not near 190K"


def test_T9_matrix_param_ratio():
    from kronllm.config import experiment_config
    from kronllm.model import KronTransformer
    e0 = KronTransformer(experiment_config("E0"))
    e2 = KronTransformer(experiment_config("E2"))
    ratio = e2.n_params_matrix / e0.n_params_matrix
    assert ratio <= 0.25, f"E2/E0 matrix params ratio {ratio:.3f} > 0.25"


# ---------------------------------------------------------------------------
# T10 -- forward/backward stability
# ---------------------------------------------------------------------------
def test_T10_no_nan(cfg):
    from kronllm.model import KronTransformer
    torch.manual_seed(0)
    m = KronTransformer(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(10):
        x = torch.randint(0, 256, (2, 32))
        y = torch.randint(0, 256, (2, 32))
        loss = m(x, y)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        opt.step()
        opt.zero_grad()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# extra sanity: effective rank helper
# ---------------------------------------------------------------------------
def test_effective_rank_bounds():
    s = torch.ones(64)
    assert abs(float(effective_rank(s)) - 64.0) < 1e-4
    s = torch.zeros(64)
    s[0] = 1.0
    assert abs(float(effective_rank(s)) - 1.0) < 1e-4


def test_kron_materialize_shape_and_rank():
    A = torch.eye(8)
    B = torch.eye(16)
    W = kron_materialize(A, B)
    assert W.shape == (128, 128)
    assert torch.allclose(W, torch.eye(128))


# ---------------------------------------------------------------------------
# T16 -- batched prelude (KronBank) must equal the per-module reference path
# ---------------------------------------------------------------------------
def _tiny_model_cfg(n_layers: int = 3, d: int = 64, n1: int = 8, n2: int = 8):
    from kronllm.config import experiment_config
    cfg = experiment_config("E2")
    cfg["data"]["seq_len"] = 32
    cfg["model"].update({"d_model": d, "n_layers": n_layers, "n_heads": 4})
    cfg["model"]["kron"].update({"n1": n1, "n2": n2})
    return cfg


def test_T16_bank_matches_reference_forward():
    from kronllm.model import KronTransformer
    torch.manual_seed(0)
    m = KronTransformer(_tiny_model_cfg())
    assert m._banks, "expected the batched prelude to be active"
    x = torch.randint(0, 256, (2, 20))
    with torch.no_grad():
        a = m(x)["logits"].clone()
        m._banks = []
        b = m(x)["logits"].clone()
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()


def test_T16_bank_matches_reference_gradients():
    from kronllm.model import KronTransformer
    torch.manual_seed(0)
    m = KronTransformer(_tiny_model_cfg())
    x = torch.randint(0, 256, (2, 20))
    y = torch.roll(x, -1, dims=1)

    loss = m(x, y)["loss"]
    loss.backward()
    g_bank = {k: p.grad.detach().clone() for k, p in m.named_parameters()
              if p.grad is not None}
    m.zero_grad(set_to_none=True)

    m._banks = []
    loss2 = m(x, y)["loss"]
    loss2.backward()
    assert abs(float(loss) - float(loss2)) < 1e-6
    for k, p in m.named_parameters():
        if k not in g_bank:
            continue
        assert p.grad is not None, k
        assert torch.allclose(g_bank[k], p.grad, atol=1e-5, rtol=1e-4), k


# ---------------------------------------------------------------------------
# T17 -- ReZero gate: starts as a small perturbation, still gets gradient
# ---------------------------------------------------------------------------
def test_T17_gate_is_active_and_differentiable():
    torch.manual_seed(0)
    m = KronCompose(64, 64, 8, 8, L=4, K=2, K_win=2, gate_init=0.1)
    assert abs(float(m.gate) - 0.1) < 1e-6
    x = torch.randn(2, 8, 64, requires_grad=True)
    y, _ = m(x, None, 0)
    y.pow(2).sum().backward()
    assert m.gate.grad is not None and abs(float(m.gate.grad)) > 0.0
    # the branch must be visible to A even though the gate is small
    assert m.A.grad is not None and float(m.A.grad.abs().sum()) > 0.0
    # branch output must be small relative to the residual stream at init
    assert float(y.norm()) < 0.5 * float(x.norm())
