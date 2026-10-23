"""Hugging Face ``transformers`` integration for KronLLM.

The wrapper exposes the custom Kronecker model through the standard
``PreTrainedModel`` / ``PreTrainedConfig`` interfaces, so that it can be saved,
loaded, moved between devices and sampled with ``GenerationMixin.generate``::

    from kronllm.hf import KronForCausalLM

    model = KronForCausalLM.from_kron_checkpoint("runs/Q2/best.pt")
    model.generate(input_ids, max_new_tokens=32, do_sample=True, top_k=8)

It also registers ``KronConfig`` and ``KronForCausalLM`` with the ``AutoClass``
registry, which makes ``AutoModelForCausalLM.from_pretrained`` work after this
module has been imported.

Two deliberate limitations are inherited from the base model:

* there is no KV cache, so every generated token is a full forward pass;
* ``attention_mask`` is accepted for API compatibility but only an all-ones
  mask is supported.  Batch generation with left/right padding would attend to
  pad tokens.  Use one sequence per call if exact masking matters.

Tokenisation is not part of this wrapper: KronLLM uses a byte-level BPE stored
in ``data/tokenizer.json`` (see :func:`kronllm.tokenizer.load_tokenizer`), which
is not a ``PreTrainedTokenizer``.  Encode text with that tokenizer, call
``generate``, then decode the result.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from transformers import (AutoConfig, AutoModelForCausalLM, PretrainedConfig,
                          PreTrainedModel)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .model import KronTransformer
from .quantization import load_any_checkpoint

__all__ = ["KronConfig", "KronForCausalLM"]


class KronConfig(PretrainedConfig):
    """``PretrainedConfig`` carrying the nested KronLLM configuration.

    Args:
        kron_config: The dictionary returned by
            :func:`kronllm.config.experiment_config` or stored inside a
            checkpoint.  It is deep-copied under the ``kron_config`` key and
            converted back by :meth:`to_kron_config`.
        **kwargs: Standard ``PretrainedConfig`` keyword arguments.
    """

    model_type = "kronllm"

    def __init__(self, kron_config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # ``PretrainedConfig.to_diff_dict`` instantiates the class without
        # arguments, so an empty config must be constructible; the model
        # wrapper rejects it later with a clearer error.
        self.kron_config = copy.deepcopy(kron_config) if kron_config else {}

        data = self.kron_config.get("data", {})
        eot = data.get("eot_id")
        if getattr(self, "vocab_size", None) is None:
            self.vocab_size = data.get("vocab_size")
        if self.eos_token_id is None:
            self.eos_token_id = eot
        if self.pad_token_id is None:
            self.pad_token_id = eot
        self.max_position_embeddings = int(data.get("seq_len", 2048))
        self.architectures = ["KronForCausalLM"]
        # The model has no KV cache; telling GenerationMixin this keeps the
        # sampling loop from passing past_key_values it cannot use.
        self.use_cache = False

    @classmethod
    def from_kron_config(cls, cfg: Dict[str, Any], **kwargs: Any) -> "KronConfig":
        """Build a config from a plain nested KronLLM config dictionary."""
        return cls(kron_config=cfg, **kwargs)

    def to_kron_config(self) -> Dict[str, Any]:
        """Return a deep copy of the nested config expected by the model.

        Raises:
            ValueError: If the config was created without a ``kron_config``.
        """
        if not self.kron_config:
            raise ValueError("KronConfig carries no 'kron_config' dictionary")
        return copy.deepcopy(self.kron_config)


class KronForCausalLM(PreTrainedModel, GenerationMixin):
    """A ``PreTrainedModel`` wrapper around :class:`kronllm.model.KronTransformer`.

    The wrapped network lives in the ``kron`` attribute, so a plain checkpoint
    saved by ``kronllm.train`` can be loaded with
    :meth:`from_kron_checkpoint`.  Weights saved with ``save_pretrained`` keep
    the same ``kron.*`` key prefix and reload through the usual
    ``from_pretrained`` path.
    """

    config_class = KronConfig
    base_model_prefix = "kron"
    # KronTransformer ties the output head to the embedding when the embedding
    # is a plain table; declare it so save_pretrained drops the duplicate.
    _tied_weights_keys = ["kron.lm_head.weight"]
    # ... and do not warn when that duplicate is absent from a checkpoint.
    _keys_to_ignore_on_load_missing = ["kron.lm_head.weight"]

    def __init__(self, config: KronConfig) -> None:
        super().__init__(config)
        if not config.kron_config:
            raise ValueError("KronForCausalLM requires a KronConfig with 'kron_config'")
        self.kron = KronTransformer(config.to_kron_config())
        self.vocab_size = self.kron.vocab_size
        # The inner model already initialises every module (including the
        # orthogonal Kronecker factors), so HF's default initialiser must not
        # touch the weights a second time.
        self.post_init()

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Leave the weights as initialised by :class:`KronTransformer`."""

    def tie_weights(self) -> None:
        """Re-tie the output head after Hugging Face replaces parameters.

        ``from_pretrained`` may assign loaded tensors with ``assign=True``,
        which breaks the shared storage created in ``KronTransformer.__init__``
        for the plain (non-factorised) embedding.  Restoring the link keeps the
        head and the embedding identical, as in training.
        """
        kron = getattr(self, "kron", None)
        if kron is None or getattr(kron, "embed_up", None) is not None:
            return
        if getattr(kron, "lm_head", None) is not None:
            kron.lm_head.weight = kron.embed.weight

    @property
    def seq_len(self) -> int:
        """Maximum context length used by the sliding-window generation loop."""
        return self.kron.seq_len

    def can_generate(self) -> bool:
        """KronLLM supports ``generate`` even though it has no KV cache."""
        return True

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run the model and optionally compute the next-token loss.

        Args:
            input_ids: ``(batch, seq)`` integer token ids.
            attention_mask: Accepted for API compatibility; only an all-ones
                mask is supported (the model has no padding-aware attention).
            labels: ``(batch, seq)`` next-token targets.  Positions equal to
                ``-100`` are ignored.  The labels are shifted internally, as in
                every Hugging Face causal language model.

        Returns:
            A :class:`CausalLMOutputWithPast` with ``logits`` of shape
            ``(batch, seq, vocab_size)`` and, when ``labels`` is given, the
            scalar cross-entropy ``loss``.
        """
        if input_ids is None:
            raise ValueError("input_ids is required")

        out = self.kron(input_ids)
        logits = out["logits"]

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor,
                                      **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Recompute from a window of at most ``seq_len`` tokens.

        KronLLM has no KV cache and was trained on fixed-length blocks, so the
        sampling loop mirrors :meth:`KronTransformer.generate` and slides a
        window over the context.
        """
        if input_ids.shape[1] > self.seq_len:
            input_ids = input_ids[:, -self.seq_len:]
        return {"input_ids": input_ids}

    @classmethod
    def from_kron_checkpoint(cls, path: str | Path,
                             **kwargs: Any) -> "KronForCausalLM":
        """Load a checkpoint written by ``kronllm.train`` or ``quantize``.

        Args:
            path: Path to ``best.pt`` / ``last.pt`` / ``model_int8.pt``.
            **kwargs: Extra arguments for ``KronConfig``.

        Returns:
            An evaluation-mode :class:`KronForCausalLM` on CPU; call ``.to()``
            to move it to a GPU.
        """
        cfg, state = load_any_checkpoint(path)
        if cfg is None:
            raise ValueError(f"{path} carries no config; pass a KronConfig explicitly")
        model = cls(KronConfig.from_kron_config(cfg, **kwargs))
        model.kron.load_state_dict(state)
        model.eval()
        return model


# Make the AutoClasses work as soon as this module is imported.
AutoConfig.register(KronConfig.model_type, KronConfig)
AutoModelForCausalLM.register(KronConfig, KronForCausalLM)
