"""Measure real throughput and the first 200 steps of the 10M configuration.

Uses the TinyStories BPE-4096 bins (the only tokenised data on disk until the
bilingual corpus is built).  Token ids are < 4096, so a vocab-8192 embedding is
fine; the point is the step time and that the optimizer actually descends at
this width, not a quality estimate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from kronllm.config import experiment_config
from kronllm.model import KronTransformer

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 24
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 512
DATA = Path("data/train.bpe4096.bin")

cfg = experiment_config("Q8g")
cfg["model"].update({"d_model": 512, "n_layers": 25, "n_heads": 8,
                     "embed_rank": 256})
cfg["model"]["kron"].update({"n1": 16, "n2": 32, "K": 10})
cfg["data"].update({"vocab_size": 8192, "seq_len": SEQ})

device = "cuda" if torch.cuda.is_available() else "cpu"
model = KronTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

tokens = np.memmap(DATA, dtype=np.uint16, mode="r")
n_seq = (len(tokens) - 1) // SEQ
rng = np.random.default_rng(0)


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, n_seq, size=BATCH) * SEQ
    x = np.stack([tokens[s:s + SEQ] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1:s + 1 + SEQ] for s in starts]).astype(np.int64)
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

t0 = time.time()
losses: list[float] = []
for step in range(1, STEPS + 1):
    x, y = batch()
    out = model(x, targets=y)
    loss = out["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    losses.append(float(loss))
    if step % 20 == 0:
        recent = sum(losses[-20:]) / 20
        print(f"step {step:3d} | loss {recent:.4f} | "
              f"{time.time()-t0:.1f}s elapsed", flush=True)
if device == "cuda":
    torch.cuda.synchronize()
dt = time.time() - t0

per_step = BATCH * SEQ
tps = STEPS * per_step / dt
peak = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
print(f"params {n_params:,} | batch {BATCH} x seq {SEQ} | "
      f"{per_step} tok/step")
print(f"{tps:.0f} tokens/s | {dt/STEPS:.2f} s/step | peak {peak:.2f} GB")
print(f"2B tokens: {2e9/tps/3600:.1f} h | 1.8B: {1.8e9/tps/3600:.1f} h")
