"""Who holds the fixed training memory?  A snapshot with allocation stacks.

`probe_vram` measured the shape of the cost -- peak scales with d^2 in width and
linearly in layers, and is the same at batch 1 and 2, so it is not activation
memory -- but not its owner.  Guessing between "the composed d x d maps and
their autograd graph" and "the depth state D of every layer" would cost a day of
work on the wrong one.

This records every CUDA allocation with its Python stack, runs one fwd+bwd at
the real training shape, and aggregates the live blocks by the first frame inside
`kronllm/`.  That frame is the line that allocated it.

Usage::

    python scripts/probe_memory_sites.py --d 512 --layers 25 --batch 1
    python scripts/probe_memory_sites.py --mode sequential      # no d x d maps
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from kronllm.config import experiment_config      # noqa: E402
from kronllm.model import KronTransformer         # noqa: E402


def build(d: int, layers: int, heads: int, kron: dict | None) -> "KronTransformer":
    cfg = experiment_config("Q10")
    m = cfg["model"]
    m["d_model"], m["n_layers"], m["n_heads"] = d, layers, heads
    m["loss_chunk"] = 1024
    if kron:
        m["kron"].update(kron)
    return KronTransformer(cfg)


def site(frames: list[dict]) -> str:
    for f in frames:
        fn = f.get("filename", "")
        if "kronllm" in fn:
            return f"{Path(fn).name}:{f['line']} {f.get('name', '')}"
    if frames:
        f = frames[0]
        return f"<{Path(f.get('filename', '?')).name}:{f.get('line')}>"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=25)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--kron", default="", help="JSON overrides for model.kron")
    ap.add_argument("--mode", default="", choices=["", "auto", "matrix",
                                                   "sequential"],
                    help="model.force_mode: 'sequential' skips the d x d map")
    ap.add_argument("--no-bank", action="store_true",
                    help="drop the batched prelude, build P per layer instead")
    ap.add_argument("--steps", type=int, default=0,
                    help="time N fwd+bwd steps instead of the memory snapshot")
    ap.add_argument("--tag", default="", help="label for the output rows")
    ap.add_argument("--out", default="data/memory_sites.json")
    args = ap.parse_args()

    kron = json.loads(args.kron) if args.kron else {}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if not args.steps:
        torch.cuda.memory._record_memory_history(max_entries=200000,
                                                 context="all")

    model = build(args.d, args.layers, args.heads, kron or None).cuda().train()
    model.force_mode = args.mode or None
    if args.no_bank:
        model.use_bank = False
        model._banks = []
    x = torch.randint(0, 8192, (args.batch, args.seq), device="cuda")
    y = torch.randint(0, 8192, (args.batch, args.seq), device="cuda")

    if args.steps:
        for _ in range(2):                       # warm up the allocator
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(x, y)
            out["loss"].backward()
            del out                              # keep the graph off the heap
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        n = args.steps
        for _ in range(n):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(x, y)
            out["loss"].backward()
            del out
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        sec = (time.perf_counter() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{args.tag or (args.mode or 'auto') + (' no-bank' if args.no_bank else '')}"
              f": {sec:.3f} s/step  peak {peak:.2f} GB")
        return

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x, y)
    fwd = torch.cuda.memory_allocated() / 1e9
    # Snapshot while the graph is alive: after the backward everything but the
    # parameters is gone and the answer is 0.1 GB.
    snap = torch.cuda.memory_snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    out["loss"].backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    live = torch.cuda.memory_allocated() / 1e9

    buckets: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for seg in snap:
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            key = site(blk.get("frames") or [])
            buckets[key][0] += int(blk.get("size", 0))
            buckets[key][1] += 1
    rows = sorted(({"site": k, "gb": round(v[0] / 1e9, 3), "blocks": v[1]}
                   for k, v in buckets.items()),
                  key=lambda r: -r["gb"])

    print(f"d={args.d} L={args.layers} batch={args.batch} seq={args.seq} "
          f"mode={args.mode or 'auto'}")
    print(f"  allocated after fwd {fwd:.2f} GB | peak {peak:.2f} GB | "
          f"live after bwd {live:.2f} GB")
    print("  top allocation sites (live bytes after backward):")
    for r in rows[:12]:
        print(f"    {r['gb']:6.3f} GB  x{r['blocks']:<5d} {r['site'][:88]}")
    tail = sum(r["gb"] for r in rows[12:])
    print(f"    {tail:6.3f} GB  ({len(rows) - 12} smaller sites)")

    Path(args.out).write_text(json.dumps(
        {"d": args.d, "layers": args.layers, "batch": args.batch,
         "seq": args.seq, "mode": args.mode or "auto",
         "fwd_gb": round(fwd, 3), "peak_gb": round(peak, 3),
         "live_gb": round(live, 3), "sites": rows}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
