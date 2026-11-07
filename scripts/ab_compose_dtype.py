"""Does keeping the composed maps in fp16 cost quality?  Two arms, one seed.

The change is numerical, not architectural: `rms_norm_matrix` used to return
fp32 under autocast (`torch.div` is one of the ops autocast widens), which
dragged the whole depth state -- (150, 4, 512, 512) at Q10 -- into fp32.  Both
arms train in fp16 so the comparison is real; only the dtype of the composition
differs:

* A: ``compose_dtype=fp32`` -- the state stays fp32, i.e. yesterday's numerics.
* B: ``compose_dtype=fp16`` -- the state stays in the activation dtype.

The bench config is fp32 by default (``training.amp: none``), where the change
does nothing at all, which is why ``amp=fp16`` is forced here.

Run more than one seed (``--seed``) before believing the delta: the first seed
came out at +0.0006 against a seed-noise floor of 0.0005, which is not a result.

Usage::

    python scripts/ab_compose_dtype.py --seed 43
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Icold\Desktop\kron_llm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = [("A_compose_fp32", "fp32"), ("B_compose_fp16", "fp16")]
    rows = []
    for name, dtype in arms:
        run_dir = ROOT / "runs" / f"ABc_{name}_s{args.seed}"
        cmd = [sys.executable, "-u", "-m", "kronllm.cli", "train",
               "--exp", "L1", "--run-dir", str(run_dir),
               "--steps", str(args.steps),
               "--override", "logging.tensorboard=false",
               "--override", f"seed={args.seed}",
               "--override", "training.batch_size=32",
               "--override", "training.warmup_steps=100",
               "--override", "training.amp=fp16",
               "--override", f"model.kron.compose_dtype={dtype}"]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        (run_dir / "train.log").write_text((r.stdout or "") + (r.stderr or ""),
                                           encoding="utf-8")
        res = run_dir / "results.json"
        bpb = (json.loads(res.read_text(encoding="utf-8"))["val_bpb"]
               if res.exists() else None)
        sec = round(time.time() - t0, 1)
        rows.append({"arm": name, "compose_dtype": dtype, "seed": args.seed,
                     "val_bpb": bpb, "sec": sec})
        print(f"{name:16s} seed {args.seed}  bpb {bpb}  ({sec:.0f} s)", flush=True)

    out = Path(args.out) if args.out else ROOT / "runs" / f"AB_compose_dtype_s{args.seed}.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    base = rows[0]["val_bpb"]
    if base and rows[1]["val_bpb"]:
        print(f"delta (fp16 - fp32) {rows[1]['val_bpb'] - base:+.4f}")


if __name__ == "__main__":
    main()
