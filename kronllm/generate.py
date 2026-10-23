"""Text generation helpers (byte-level and BPE)."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch

from .model import KronTransformer
from .tokenizer import ByteTokenizer


def encode(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="ignore"))


def decode(ids: Sequence[int]) -> str:
    return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


@torch.no_grad()
def generate_texts(model: KronTransformer, prompts: Iterable[str],
                   max_new_tokens: int = 200, temperature: float = 1.0,
                   top_k: Optional[int] = 8, n_samples: int = 1,
                   seed: Optional[int] = None,
                   device: torch.device | str | None = None,
                   stop_at_eot: bool = True,
                   tokenizer=None) -> List[str]:
    """Sample from ``model``; ``tokenizer`` defaults to the byte vocabulary."""
    tok = tokenizer if tokenizer is not None else ByteTokenizer()
    device = torch.device(device) if device is not None else model.device
    prompts = [p if p else "\n" for p in prompts] or ["Once upon a time"]
    if seed is not None:
        torch.manual_seed(int(seed))
    texts: List[str] = []
    for i in range(max(1, n_samples)):
        prompt = prompts[i % len(prompts)]
        ids = tok.encode(prompt) or [tok.eot_id]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k,
                             eot_id=tok.eot_id, stop_at_eot=stop_at_eot)
        texts.append(tok.decode(out[0].tolist()))
    return texts
