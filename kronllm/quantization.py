"""Post-training quantization for KronLLM checkpoints.

KronLLM does not contain any ``nn.Linear`` layer: every weight matrix is a
Kronecker composition of small factors, so ``torch.ao.quantization`` cannot see
it.  Instead we quantize the *stored* tensors -- factors, embeddings and the
tied head -- with symmetric per-channel scales and dequantize them when the
checkpoint is loaded.  The forward pass is therefore bit-identical in structure
and only sees float weights; the win is a 4x (int8) or 8x (int4) smaller file,
which is what makes the model practical on a phone or a microcontroller.

Typical use::

    from kronllm.quantization import quantize_checkpoint, load_any_checkpoint

    report = quantize_checkpoint("runs/Q2/best.pt", "runs/Q2/model_int8.pt", bits=8)
    cfg, state = load_any_checkpoint("runs/Q2/model_int8.pt")
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

# Tensors smaller than this are stored in float32: scales would cost more than
# the payload, and 1-D parameters (RMSNorm, tau, gate) must stay exact.
_MIN_NUMEL = 256

# Name fragments that must never be quantized (normalisation and scalars).
_SKIP_FRAGMENTS = ("norm", "tau", "gate", "theta", "anchor")


def _is_quantizable(name: str, tensor: torch.Tensor) -> bool:
    """Return True if ``tensor`` should be stored in low precision.

    Args:
        name: Parameter name inside the state dict.
        tensor: Parameter tensor.

    Returns:
        True for 2-D (or higher) float tensors of at least ``_MIN_NUMEL``
        elements whose name does not match a normalisation/scalar role.
    """
    if not tensor.is_floating_point() or tensor.dim() < 2:
        return False
    if tensor.numel() < _MIN_NUMEL:
        return False
    return not any(fragment in name for fragment in _SKIP_FRAGMENTS)


def _pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 codes (range ``[-8, 7]``) two per byte.

    Args:
        codes: Int8 tensor whose values are already clamped to ``[-8, 7]``.

    Returns:
        Flat uint8 tensor with the low nibble holding the even elements.
    """
    flat = codes.reshape(-1).to(torch.int16)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])
    shifted = flat + 8                       # 0..15, safe in int16
    packed = shifted[0::2] | (shifted[1::2] << 4)
    return packed.to(torch.uint8)


def _unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of :func:`_pack_int4`.

    Args:
        packed: Flat uint8 tensor produced by :func:`_pack_int4`.
        numel: Number of original elements.

    Returns:
        Int8 tensor of length ``numel`` in ``[-8, 7]``.
    """
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(packed.numel() * 2, dtype=torch.int16)
    out[0::2] = low
    out[1::2] = high
    return out[:numel].to(torch.int8)


def _quantize_tensor(tensor: torch.Tensor, bits: int
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel quantization of one tensor.

    Args:
        tensor: Float tensor to quantize.
        bits: Target bit width, either 8 or 4.

    Returns:
        A pair ``(codes, scale)``.  ``codes`` is int8 (values clamped to the
        signed range) and ``scale`` broadcasts against ``tensor``.
    """
    qmax = float(2 ** (bits - 1) - 1)
    reduce_dims = tuple(range(1, tensor.dim()))
    amax = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = (amax / qmax).clamp_min(1e-8)
    codes = torch.round(tensor.detach() / scale).clamp(-qmax, qmax).to(torch.int8)
    return codes, scale


def quantize_state_dict(state: Mapping[str, torch.Tensor], bits: int = 8
                        ) -> Dict[str, Any]:
    """Quantize every eligible tensor of a state dict.

    Args:
        state: Float state dict as stored in a training checkpoint.
        bits: 8 or 4.

    Returns:
        A serialisable dict mapping parameter names to either a float tensor
        (kept exact) or a ``{"q", "scale", "shape"}`` record.
    """
    if bits not in (8, 4):
        raise ValueError(f"bits must be 8 or 4, got {bits}")

    out: Dict[str, Any] = {}
    for name, tensor in state.items():
        if not _is_quantizable(name, tensor):
            out[name] = tensor.detach().clone()
            continue
        codes, scale = _quantize_tensor(tensor, bits)
        record: Dict[str, Any] = {"shape": tuple(tensor.shape), "scale": scale}
        if bits == 8:
            record["q"] = codes.reshape(tensor.shape)
        else:
            record["q"] = _pack_int4(codes)
            record["numel"] = tensor.numel()
        out[name] = record
    return out


def dequantize_state_dict(quantized: Mapping[str, Any], bits: int
                          ) -> Dict[str, torch.Tensor]:
    """Rebuild a float state dict from :func:`quantize_state_dict` output.

    Args:
        quantized: Output of :func:`quantize_state_dict`.
        bits: Bit width used during quantization.

    Returns:
        Float state dict that can be passed to ``model.load_state_dict``.
    """
    state: Dict[str, torch.Tensor] = {}
    for name, value in quantized.items():
        if not isinstance(value, Mapping):
            state[name] = value.float() if value.is_floating_point() else value
            continue
        scale = value["scale"].float()
        if bits == 8:
            codes = value["q"].float()
        else:
            codes = _unpack_int4(value["q"], int(value["numel"])).float()
            codes = codes.reshape(value["shape"])
        state[name] = codes * scale
    return state


def quantize_checkpoint(src: str | Path, dst: str | Path, bits: int = 8
                        ) -> Dict[str, Any]:
    """Quantize a training checkpoint and write a compact checkpoint.

    Args:
        src: Path to a checkpoint written by ``kronllm.train`` (``best.pt``).
        dst: Destination path for the quantized checkpoint.
        bits: 8 or 4.

    Returns:
        A report with the original and quantized sizes plus the parameter count.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    quantized = quantize_state_dict(state, bits=bits)

    payload = {
        "quantized": True,
        "bits": bits,
        "cfg": checkpoint.get("cfg"),
        "step": checkpoint.get("step"),
        "val_ppl": checkpoint.get("val_ppl"),
        "state": quantized,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)

    return {
        "src": str(src),
        "dst": str(dst),
        "bits": bits,
        "params": int(sum(v.numel() for v in state.values())),
        "size_before_mb": round(src.stat().st_size / 1e6, 3),
        "size_after_mb": round(dst.stat().st_size / 1e6, 3),
        "compression": round(src.stat().st_size / dst.stat().st_size, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def load_any_checkpoint(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Load a float or a quantized checkpoint.

    Args:
        path: Checkpoint path.

    Returns:
        A pair ``(cfg, state_dict)`` where ``state_dict`` is always float and
        can be loaded into :class:`kronllm.model.KronTransformer`.

    Raises:
        ValueError: If the checkpoint is quantized but carries no config.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint.get("quantized"):
        cfg = checkpoint.get("cfg")
        return cfg, checkpoint.get("model", checkpoint)

    cfg = checkpoint.get("cfg")
    if cfg is None:
        raise ValueError(f"quantized checkpoint {path} has no config")
    state = dequantize_state_dict(checkpoint["state"], int(checkpoint["bits"]))
    return cfg, state
