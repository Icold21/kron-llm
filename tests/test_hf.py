"""Tests for the Hugging Face ``PreTrainedModel`` wrapper (T27)."""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from kronllm.hf import KronConfig, KronForCausalLM  # noqa: E402
from kronllm.quantization import quantize_checkpoint  # noqa: E402
from kronllm.testutils import make_cfg  # noqa: E402


def _small_model() -> KronForCausalLM:
    return KronForCausalLM(KronConfig.from_kron_config(make_cfg())).eval()


def test_config_roundtrip() -> None:
    cfg = make_cfg()
    config = KronConfig.from_kron_config(cfg)

    assert config.model_type == "kronllm"
    assert config.to_kron_config() == cfg
    assert config.eos_token_id == 255
    assert config.pad_token_id == 255
    assert config.architectures == ["KronForCausalLM"]
    assert config.use_cache is False


def test_forward_loss_and_tied_head() -> None:
    model = _small_model()
    assert model.kron.lm_head.weight is model.kron.embed.weight

    ids = torch.randint(0, 256, (2, 12))
    out = model(input_ids=ids, labels=ids)

    assert out.logits.shape == (2, 12, 256)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert sum(1 for p in model.parameters()
               if p.grad is not None and p.grad.abs().sum() > 0) > 0


def test_save_load_and_autoclasses(tmp_path) -> None:
    model = _small_model()
    ids = torch.randint(0, 256, (2, 10))
    with torch.no_grad():
        reference = model(input_ids=ids).logits

    model.save_pretrained(tmp_path)
    reloaded = KronForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        restored = reloaded(input_ids=ids).logits

    assert torch.allclose(reference, restored, atol=1e-6)
    assert reloaded.kron.lm_head.weight is reloaded.kron.embed.weight

    config = transformers.AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, KronConfig)
    auto = transformers.AutoModelForCausalLM.from_pretrained(tmp_path)
    assert isinstance(auto, KronForCausalLM)
    with torch.no_grad():
        assert torch.allclose(reference, auto(input_ids=ids).logits, atol=1e-6)


def test_from_checkpoint_and_generate(tmp_path) -> None:
    model = _small_model()
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": model.kron.state_dict(), "cfg": make_cfg(), "step": 7},
               checkpoint)

    loaded = KronForCausalLM.from_kron_checkpoint(checkpoint)
    ids = torch.randint(0, 256, (1, 6))
    out = loaded.generate(
        ids, attention_mask=torch.ones_like(ids),
        max_new_tokens=4, do_sample=False,
    )
    assert out.shape == (1, 10)


def test_from_quantized_checkpoint(tmp_path) -> None:
    model = _small_model()
    source = tmp_path / "best.pt"
    quantized = tmp_path / "model_int8.pt"
    torch.save({"model": model.kron.state_dict(), "cfg": make_cfg(), "step": 3},
               source)
    quantize_checkpoint(source, quantized, bits=8)

    loaded = KronForCausalLM.from_kron_checkpoint(quantized)
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        logits = loaded(input_ids=ids).logits
    assert logits.shape == (1, 8, 256)
    assert torch.isfinite(logits).all()
