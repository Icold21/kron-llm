from __future__ import annotations

import pytest
import torch

from kronllm.testutils import make_cfg  # noqa: F401  (re-exported for tests)


@pytest.fixture
def cfg():
    return make_cfg()


@pytest.fixture
def model(cfg):
    from kronllm.model import KronTransformer
    torch.manual_seed(0)
    return KronTransformer(cfg)


@pytest.fixture
def kron_mod():
    from kronllm.kron import KronCompose
    torch.manual_seed(0)
    return KronCompose(64, 64, 8, 8, L=4, K=2, K_win=2, s_init=0.1,
                       anchor=True, s_learnable=False, stop_gradient=True,
                       rezero_eps=0.0)
