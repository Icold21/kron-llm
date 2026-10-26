"""Profile one Q10 training step: where does the time actually go?

Usage::

    python scripts/profile_step.py --data-dir data/bilingual --steps 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.config import experiment_config
from kronllm.data import open_dataset
from kronllm.model import KronTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bilingual")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    overrides = [
        f"data.data_dir={args.data_dir}",
        f"training.batch_size={args.batch_size}",
        "training.amp=fp16",
    ]
    cfg = experiment_config("Q10", None, overrides)
    tc = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[profile] device {device}, batch {args.batch_size}, "
          f"seq {tc['seq_len']}", flush=True)

    ds = open_dataset(cfg, "train")
    ds.set_shard(0, 1, seed=1234)
    model = KronTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(7)
    bs = args.batch_size

    def step() -> float:
        x, y = ds.get_batch(bs, device, gen)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
            out = model(x, y)
            loss = out["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    for _ in range(3):                       # warmup: allocator, cudnn
        step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False,
    )
    with prof:
        for _ in range(args.steps):
            step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    print("\n=== top CUDA ops ===", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18),
          flush=True)
    print("\n=== top CPU ops ===", flush=True)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=18),
          flush=True)

    # crude split: prelude vs attention vs ffn vs optimizer
    totals: dict[str, float] = {}
    for evt in prof.key_averages():
        name = evt.key
        cuda = getattr(evt, "cuda_time_total", 0.0)
        for tag in ("kron", "einsum", "attention", "ffn", "optimizer",
                    "elementwise", "copy", "memcpy"):
            if tag in name.lower():
                totals[tag] = totals.get(tag, 0.0) + cuda
    print("\n=== keyword totals (CUDA us) ===", flush=True)
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{k:12s} {v/1000:10.1f} ms", flush=True)


if __name__ == "__main__":
    main()
