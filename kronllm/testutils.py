"""Config helpers shared by the test suite (also usable from scripts)."""
from __future__ import annotations

from typing import Any, Dict

from .config import DEFAULTS, deep_merge


def make_cfg(**overrides: Any) -> Dict[str, Any]:
    """Small, fast config for unit/integration tests.

    ``overrides`` use dotted keys, e.g. ``{"model.d_model": 64}``.
    """
    cfg = deep_merge(DEFAULTS, {
        "data": {"seq_len": 64, "vocab_size": 256},
        "model": {
            "d_model": 64, "n_layers": 2, "n_heads": 4,
            "kron": {"n1": 8, "n2": 8},
        },
        "training": {"batch_size": 2, "grad_accum": 1, "amp": "none",
                     "device": "cpu", "warmup_steps": 5, "eval_every": 1000},
    })
    for key, value in overrides.items():
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg
