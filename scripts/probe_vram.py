"""Will a 20M model fit on a 16 GB T4?  Measure it here, in the same shapes.

Peak training memory is set by the tensors the graph holds -- activations, the
chunked-loss buffers and the fp16 master copies -- not by which card runs them,
so an fp16 forward+backward measured on this 6 GB card gives a number that
transfers to the T4, except for allocator slack.

Reports peak VRAM for one fwd+bwd at seq 512 with the chunked loss (the same
``loss_chunk`` the Q10 kernel uses), for the current shape and for the candidate
20M shapes, at several batch sizes.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import gc
import json

import torch

from kronllm.config import experiment_config
from kronllm.model import KronTransformer


def measure(d_model: int, n_layers: int, n_heads: int, batch: int,
            seq: int = 512, kron: dict | None = None) -> dict:
    cfg = experiment_config("Q10")
    m = cfg["model"]
    m["d_model"] = d_model
    m["n_layers"] = n_layers
    m["n_heads"] = n_heads
    m["loss_chunk"] = 1024
    if kron:
        m["kron"].update(kron)
    model = KronTransformer(cfg).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 8192, (batch, seq), device="cuda")
    y = torch.randint(0, 8192, (batch, seq), device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, y)
            loss = out["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        ok = True
    except torch.cuda.OutOfMemoryError:
        peak = reserved = float("nan")
        ok = False
    params = sum(p.numel() for p in model.parameters())
    del model, opt, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return {"params": params, "batch": batch, "seq": seq, "ok": ok,
            "peak_gb": round(peak, 2), "reserved_gb": round(reserved, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vram_probe.json")
    args = ap.parse_args()

    shapes = [
        ("Q10  d512 L25 h8", 512, 25, 8, None),
        ("Q11  d640 L25 h8", 640, 25, 8, {"n1": 20, "n2": 32}),
        ("Q12  d768 L25 h12", 768, 25, 12, {"n1": 24, "n2": 32}),
        ("Q12b d512 L50 h8", 512, 50, 8, None),
    ]
    rows = []
    for name, d, layers, heads, kron in shapes:
        for batch in (2, 4, 8):
            r = measure(d, layers, heads, batch, kron=kron)
            r["shape"] = name
            rows.append(r)
            print(f"{name:20s} batch {batch:2d}  {r['params']/1e6:5.2f}M  "
                  f"peak {r['peak_gb'] if r['ok'] else 'OOM':>6} GB  "
                  f"(reserved {r['reserved_gb'] if r['ok'] else '-'})", flush=True)
            if not r["ok"]:
                break
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
