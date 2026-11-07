"""The composition must stay in the dtype it is given.

The bug this guards against is invisible: `torch.div` is one of the ops autocast
widens to fp32, so a normalisation written as `x / r` silently promoted the whole
depth state to fp32 under fp16 training and cost 3.8 GB of a 5.4 GB budget.  No
loss number moves, no exception is raised; only the memory does.
"""
from __future__ import annotations

import pytest
import torch

from kronllm.kron import KronCompose, rms_norm_matrix


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rms_norm_matrix_keeps_the_input_dtype(dtype):
    x = torch.randn(3, 4, 16, 16, dtype=dtype)
    y = rms_norm_matrix(x)
    assert y.dtype is dtype
    # unit RMS per matrix, whatever the dtype
    rms = y.float().pow(2).mean(dim=(-2, -1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-2)


def test_rms_norm_matrix_matches_the_fp32_definition():
    x = torch.randn(2, 3, 8, 8, dtype=torch.float16)
    ref = x.float() / x.float().pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
    assert torch.allclose(rms_norm_matrix(x).float(), ref, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA autocast")
def test_autocast_does_not_promote_the_normalised_map():
    x = torch.randn(2, 4, 32, 32, device="cuda", dtype=torch.float16)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        y = rms_norm_matrix(x)
    assert y.dtype is torch.float16


@pytest.mark.parametrize("choice,expected", [
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp32", torch.float32),
])
def test_work_dtype_explicit_choices(choice, expected):
    mod = KronCompose(16, n1_in=4, n2_in=4, n1_out=4, n2_out=4, L=1, K=1,
                      compose_dtype=choice)
    ref = torch.zeros(1, dtype=torch.float16)
    assert mod.work_dtype(ref) is expected


def test_work_dtype_auto_follows_the_reference():
    mod = KronCompose(16, n1_in=4, n2_in=4, n1_out=4, n2_out=4, L=1, K=1)
    assert mod.work_dtype(torch.zeros(1, dtype=torch.float16)) is torch.float16
    assert mod.work_dtype(torch.zeros(1, dtype=torch.float32)) is torch.float32


def test_compose_dtype_rejects_a_typo():
    with pytest.raises(ValueError):
        KronCompose(16, n1_in=4, n2_in=4, n1_out=4, n2_out=4, L=1, K=1,
                    compose_dtype="fp8")
