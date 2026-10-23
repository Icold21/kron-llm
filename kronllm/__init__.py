"""KronLLM: a fully Kronecker-parameterised byte-level language model."""
from .kron import (KronCompose, apply_rope, effective_rank, kron_apply,
                   kron_materialize, rope_frequencies)
from .model import KronBlock, KronTransformer, RMSNorm

__all__ = [
    "KronCompose", "KronTransformer", "KronBlock", "RMSNorm",
    "kron_apply", "kron_materialize", "effective_rank",
    "apply_rope", "rope_frequencies",
]
__version__ = "2.0.0"
