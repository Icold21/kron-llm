"""Tests for post-training int8/int4 quantization."""
from __future__ import annotations

import torch

from kronllm.config import experiment_config
from kronllm.model import KronTransformer
from kronllm.quantization import (_pack_int4, _unpack_int4, dequantize_state_dict,
                                  load_any_checkpoint, quantize_checkpoint,
                                  quantize_state_dict)


def _weight(rows: int = 16, cols: int = 16) -> torch.Tensor:
    """Deterministic tensor with very different magnitudes per row."""
    codes = (torch.arange(rows * cols, dtype=torch.float32) - rows * cols / 2)
    return (codes.reshape(rows, cols) / 7.0).sin()


def test_int8_error_is_bounded_by_half_a_step() -> None:
    weight = _weight()
    state = quantize_state_dict({"block.weight": weight}, bits=8)
    restored = dequantize_state_dict(state, bits=8)["block.weight"]
    step = state["block.weight"]["scale"].max()
    assert (restored - weight).abs().max() <= step / 2 + 1e-6


def test_int4_error_is_bounded_by_half_a_step() -> None:
    weight = _weight()
    state = quantize_state_dict({"block.weight": weight}, bits=4)
    restored = dequantize_state_dict(state, bits=4)["block.weight"]
    step = state["block.weight"]["scale"].max()
    assert restored.shape == weight.shape
    assert (restored - weight).abs().max() <= step / 2 + 1e-6


def test_int4_packing_round_trip_handles_odd_lengths() -> None:
    codes = torch.randint(-8, 8, (7, 5), dtype=torch.int8)
    unpacked = _unpack_int4(_pack_int4(codes), codes.numel())
    assert torch.equal(unpacked.reshape(codes.shape), codes)


def test_small_and_one_dimensional_tensors_stay_exact() -> None:
    state = quantize_state_dict({
        "norm.weight": torch.randn(320),
        "tau": torch.randn(6),
        "tiny.weight": torch.randn(4, 4),
    }, bits=4)
    assert isinstance(state["norm.weight"], torch.Tensor)
    assert isinstance(state["tau"], torch.Tensor)
    assert isinstance(state["tiny.weight"], torch.Tensor)


def test_quantized_checkpoint_is_smaller_and_reloads(tmp_path) -> None:
    cfg = experiment_config("E0")
    model = KronTransformer(cfg)
    source = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": 7}, source)

    with torch.no_grad():
        tokens = torch.randint(0, 256, (2, 64))
        reference = model(tokens)["logits"].float()

    for bits, tolerance in ((8, 0.05), (4, 0.6)):
        destination = tmp_path / f"model_int{bits}.pt"
        report = quantize_checkpoint(source, destination, bits=bits)
        assert report["size_after_mb"] < report["size_before_mb"]

        loaded_cfg, state = load_any_checkpoint(destination)
        assert loaded_cfg["model"]["d_model"] == cfg["model"]["d_model"]
        quantized = KronTransformer(loaded_cfg)
        quantized.load_state_dict(state)
        with torch.no_grad():
            output = quantized(tokens)["logits"].float()

        relative = (output - reference).norm() / reference.norm()
        assert relative < tolerance, (bits, float(relative))
